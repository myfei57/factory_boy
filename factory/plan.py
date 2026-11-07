# Copyright: See the LICENSE file.

"""Two-phase persistence: preview an ORM create() without touching the database,
then execute (and, if needed, resume) the planned steps.

The public entry point is :meth:`factory.Factory.plan_create`, which returns a
:class:`PersistencePlan`.  Building the plan performs the *exact same*
declaration evaluation as :meth:`factory.Factory.create` (sequences, lazy
attributes, traits, sub-factories, ...), but no framework persistence code
ever runs: model instantiation and saving are recorded as
:class:`PlannedStep` objects instead.

The plan can then be executed through :meth:`PersistencePlan.commit`.  Each
step stores its execution status, so a failure leaves a precise report of what
was already completed; calling ``commit()`` again resumes the plan from the
failed step without re-creating already persisted objects.

Note: user-provided declarations (LazyFunction/LazyAttribute/Iterator values
or a custom ``_setup_next_sequence``) run exactly as during ``create()``; if
such user code performs database queries, those still happen.  factory_boy
itself issues no database access while planning.
"""

from __future__ import annotations

import contextlib
import types

from . import base, builder, declarations, enums, errors

# ---------------------------------------------------------------------------
# Step taxonomy
# ---------------------------------------------------------------------------


class StepKind:
    #: Instantiate the model and persist it (the ``_create`` hook).
    INSTANTIATE = 'instantiate'

    #: A single RelatedFactory, generated after the parent was saved.
    RELATED = 'related'

    #: A RelatedFactoryList, wrapping one RELATED step per generated object.
    RELATED_LIST = 'related_list'

    #: A PostGenerationMethodCall (``obj.set_password(...)`` and friends).
    POST_METHOD = 'post_method'

    #: A plain @post_generation hook.
    POST_HOOK = 'post_hook'

    #: A Maybe declaration whose branch was selected at planning time.
    POST_MAYBE = 'post_maybe'

    #: A post-generation slot that does nothing (Maybe with no branch / SKIP).
    POST_SKIPPED = 'post_skipped'

    #: A post-generation slot filled with a flat value.
    POST_VALUE = 'post_value'

    #: A custom PostGenerationDeclaration replayed verbatim at execution.
    POST_GENERIC = 'post_generic'

    #: The _after_postgeneration hook (Django's deprecated re-save lives here).
    FINALIZE = 'finalize'


class StepStatus:
    PENDING = 'pending'
    DONE = 'done'
    FAILED = 'failed'
    SKIPPED = 'skipped'


class TransactionBoundary:
    """Description of a transaction boundary around one or more steps.

    Attributes:
        kind (str): one of:

            - ``autocommit``: each statement commits on its own (Django default);
            - ``atomic``: an explicit atomic block opened by the ORM
              (get_or_create);
            - ``commit``: a session.commit() at the end of the step;
            - ``flush``: a session.flush() synchronization point inside a
              caller-owned transaction;
            - ``managed``: the caller owns the transaction, factory_boy never
              flushes nor commits.
        label (str): human-readable description.
        commits (bool): whether persistence becomes durable at boundary close.
        steps (list[PlannedStep]): steps enclosed in the boundary, in order.
    """

    __slots__ = ('kind', 'label', 'commits', 'steps')

    def __init__(self, kind, label, commits=False):
        self.kind = kind
        self.label = label
        self.commits = commits
        self.steps = []

    def __repr__(self):
        return '<TransactionBoundary: %s (%s)>' % (self.label, self.kind)


# ---------------------------------------------------------------------------
# Plan nodes
# ---------------------------------------------------------------------------


class PlannedInstance:
    """Placeholder standing in for a model instance while planning.

    It exposes the *resolved* attributes of the future instance, so that
    ``SelfAttribute``/``LazyAttribute`` lookups in sibling declarations behave
    exactly as on a real build.  Placeholders are swapped for the actual
    instance during execution.
    """

    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __getattr__(self, name):
        try:
            attributes = self.node.attributes
        except AttributeError:
            raise AttributeError(name)
        if name in attributes:
            return attributes[name]
        if name in self.node.kwargs:
            return self.node.kwargs[name]
        raise AttributeError(name)

    def __repr__(self):
        return '<PlannedInstance for %s (step #%s)>' % (
            getattr(self.node.model, '__name__', self.node.model),
            self.node.id,
        )


class PlannedStep:
    """One node of a :class:`PersistencePlan`.

    Steps form a tree:

    - :attr:`pre_children`: relations that must exist before the model is
      instantiated (SubFactory chains), in evaluation order;
    - the step itself (instantiation or a post-generation action);
    - :attr:`post_children`: post-generation actions (RelatedFactory chains,
      hooks, the finalize/re-save step), in declaration order.
    """

    def __init__(self, step_id, kind, *, name='', declaration_name=None,
                 factory=None, model=None, boundary=None):
        self.id = step_id
        self.kind = kind
        self.name = name
        self.declaration_name = declaration_name
        self.factory = factory
        self.model = model

        # Instantiation payload (kind == INSTANTIATE)
        self.attributes = {}
        self.args = ()
        self.kwargs = {}
        #: Raw call kwargs; used by Django/SQLAlchemy get_or_create recovery.
        self.original_params = {}

        # Post-generation payload
        self.method_name = None
        self.method_args = ()
        self.method_kwargs = {}
        self.hook_function = None
        self.hook_context = None
        self.declaration = None
        self.overrides = None
        self.provided_value = None

        # Tree
        self.pre_children = []
        self.post_children = []
        #: For RELATED steps, the child instantiation step (None when a value
        #: was provided instead of generating the related object).
        self.related_child = None
        #: Kwarg name under which the parent reaches the related factory.
        self.related_name = ''

        # Persistence / execution
        self.boundary = boundary
        self.muted_signals = ()
        self.status = StepStatus.PENDING
        self.result = None
        self.error = None
        #: Whether the model row was actually instantiated/persisted.
        #: Unlike :attr:`status`, this stays True if a later post hook fails,
        #: so a retry resumes the hooks without inserting a duplicate row.
        self.persisted = False

    # -- status helpers ----------------------------------------------------

    @property
    def done(self):
        return self.status == StepStatus.DONE

    @property
    def failed(self):
        return self.status == StepStatus.FAILED

    @property
    def pending(self):
        return self.status == StepStatus.PENDING

    @property
    def skipped(self):
        return self.status == StepStatus.SKIPPED

    # -- traversal ---------------------------------------------------------

    def iter_flat(self):
        """Yield steps in execution order (depth-first, pre-relations first)."""
        for child in self.pre_children:
            yield from child.iter_flat()
        yield self
        for post in self.post_children:
            yield from _iter_post_flat(post)

    def _brief(self):
        target = getattr(self.model, '__name__', self.model)
        if self.kind == StepKind.INSTANTIATE:
            return 'instantiate+save %s' % target
        if self.kind == StepKind.RELATED:
            if self.related_child is None:
                return 'related %r: use provided value' % self.declaration_name
            return 'related %r -> %s' % (
                self.declaration_name,
                getattr(self.related_child.model, '__name__', '?'))
        if self.kind == StepKind.RELATED_LIST:
            return 'related list %r (%d item(s))' % (
                self.declaration_name, len(self.post_children))
        if self.kind == StepKind.POST_METHOD:
            return 'post_generation %s.%s()' % (target, self.method_name)
        if self.kind == StepKind.POST_HOOK:
            return 'post_generation hook %s' % getattr(
                self.hook_function, '__name__', self.hook_function)
        if self.kind == StepKind.POST_MAYBE:
            return 'maybe %r' % self.declaration_name
        if self.kind == StepKind.FINALIZE:
            return 'finalize %s (_after_postgeneration)' % target
        return self.name or self.kind

    def describe(self, indent=0):
        prefix = '  ' * indent
        lines = ['%s#%s [%s] %s' % (
            prefix, self.id, self.status, self._brief())]
        for child in self.pre_children:
            lines.append(child.describe(indent + 1))
        for post in self.post_children:
            lines.append(post.describe(indent + 1))
        return '\n'.join(lines)

    def __repr__(self):
        return '<PlannedStep #%s %s: %s>' % (
            self.id, self.kind, self._brief())


def _embedded_instances(value, out):
    """Collect PlannedInstance nodes embedded in post hook payloads."""
    if isinstance(value, PlannedInstance):
        node = value.node
        if node not in out:
            out.append(node)
    elif isinstance(value, dict):
        for sub in value.values():
            _embedded_instances(sub, out)
    elif isinstance(value, (list, tuple, set)):
        for sub in value:
            _embedded_instances(sub, out)


def _iter_post_flat(post):
    """Flatten a post-generation node (which wraps its generated children)."""
    # Relations hidden inside hook arguments are built before the hook runs.
    embedded = []
    _embedded_instances(post.method_args, embedded)
    _embedded_instances(post.method_kwargs, embedded)
    if post.hook_context is not None:
        _embedded_instances(post.hook_context.value, embedded)
        _embedded_instances(post.hook_context.extra, embedded)
    _embedded_instances(post.provided_value, embedded)
    for node in embedded:
        yield from node.iter_flat()

    yield post
    if post.kind == StepKind.RELATED and post.related_child is not None:
        yield from post.related_child.iter_flat()
    elif post.kind == StepKind.RELATED_LIST:
        for child_wrapper in post.post_children:
            yield from _iter_post_flat(child_wrapper)
    elif post.kind == StepKind.POST_MAYBE:
        for child in post.post_children:
            yield from _iter_post_flat(child)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PartialCommitError(errors.FactoryError):
    """Raised when executing a :class:`PersistencePlan` failed midway.

    Attributes:
        plan (PersistencePlan): the plan, with per-step statuses filled in.
        failed_step (PlannedStep): the step that raised.
        completed (list[PlannedStep]): steps successfully executed, in order.
        remaining (list[PlannedStep]): steps not executed yet.
        original (Exception): the underlying error (also ``self.__cause__``).
    """

    def __init__(self, plan, failed_step, original):
        self.plan = plan
        self.failed_step = failed_step
        self.completed = [s for s in plan.steps if s.status == StepStatus.DONE]
        self.remaining = [s for s in plan.steps if s.status == StepStatus.PENDING]
        self.original = original
        super().__init__(
            'Execution of persistence plan failed at step #%s (%s): %r. '
            '%d step(s) completed; fix the issue and call plan.commit() again '
            'to resume idempotently.' % (
                failed_step.id, failed_step._brief(), original,
                len(self.completed),
            )
        )


# ---------------------------------------------------------------------------
# Planning context (shadow sequence counters, boundaries, ...)
# ---------------------------------------------------------------------------


class _PlanContext:
    def __init__(self, root_factory):
        self.root_factory = root_factory
        self._step_ids = 0
        self._sequences = {}
        self._metas_seen = set()
        self._iterators = []
        self._boundaries = []
        self._sa_shared_boundary = None
        #: Pre-relation sink: instantiation nodes whose SubFactory chains are
        #: currently being resolved attach here. None while planning posts.
        self._pre_sink = None
        self.roots = []

    def next_step_id(self):
        self._step_ids += 1
        return self._step_ids

    # -- sequence isolation ------------------------------------------------

    def next_sequence(self, meta, force_sequence=None, force_init_sequence=None):
        if force_sequence is not None:
            return force_sequence
        if force_init_sequence is not None:
            return force_init_sequence
        counter_ref = meta.counter_reference
        if counter_ref not in self._sequences:
            # Read the counter's current value without advancing it, so that
            # a plan previews the same sequence values a create() would use
            # while leaving the factory's global counter untouched.
            counter_ref._initialize_counter()
            self._sequences[counter_ref] = counter_ref._counter.seq
        value = self._sequences[counter_ref]
        self._sequences[counter_ref] += 1
        return value

    def note_meta(self, meta):
        if meta in self._metas_seen:
            return
        self._metas_seen.add(meta)
        for declaration_set in (meta.pre_declarations, meta.post_declarations):
            for entry in list(declaration_set.declarations.values()):
                self._collect_iterators(entry)
            for sub in declaration_set.contexts.values():
                for value in sub.values():
                    self._collect_iterators(value)

    def _collect_iterators(self, value):
        if isinstance(value, declarations.Iterator):
            if value not in self._iterators:
                self._iterators.append(value)
        elif isinstance(value, declarations.RelatedFactory):
            for sub in value.defaults.values():
                self._collect_iterators(sub)
        elif isinstance(value, declarations.PostGenerationMethodCall):
            for sub in value.method_kwargs.values():
                self._collect_iterators(sub)

    def reset_iterators(self):
        # Restore any Iterator advanced while planning so that plan_create()
        # has no observable effect on factory state.
        seen = set()
        for iterator_decl in self._iterators:
            if (iterator_decl.iterator is not None
                    and id(iterator_decl) not in seen):
                seen.add(id(iterator_decl))
                iterator_decl.reset()

    # -- transaction boundaries --------------------------------------------

    def _new_boundary(self, kind, label, commits=False):
        boundary = TransactionBoundary(kind, label, commits=commits)
        self._boundaries.append(boundary)
        return boundary

    def attach(self, node, kind, *, factory_name='', method_name=''):
        """Compute and register the boundary of a node, ORM-agnostically."""
        meta = node.factory._meta

        # SQLAlchemy options
        if hasattr(meta, 'sqlalchemy_session_persistence'):
            persistence = meta.sqlalchemy_session_persistence
            if kind == StepKind.INSTANTIATE:
                if persistence == 'commit':
                    # A committed row closes the current session transaction.
                    self._sa_shared_boundary = None
                    node.boundary = self._new_boundary(
                        'commit',
                        'session.commit() after persisting %s' % factory_name,
                        commits=True,
                    )
                elif persistence == 'flush':
                    node.boundary = self._sa_shared(
                        'flush',
                        'caller transaction; session.flush() after each %s'
                        % factory_name,
                    )
                else:
                    node.boundary = self._sa_shared(
                        'managed',
                        'caller-owned transaction (session.add only) for %s'
                        % factory_name,
                    )
            else:
                # Hooks join the currently open session transaction; after a
                # commit-persistence node a new one starts implicitly.
                node.boundary = self._sa_shared(
                    'managed',
                    'SQLAlchemy session transaction around %s'
                    % (method_name or 'post-generation hooks'))
            node.boundary.steps.append(node)
            return

        # Django options
        if hasattr(meta, 'django_get_or_create'):
            alias = getattr(meta, 'database', 'default')
            if kind == StepKind.INSTANTIATE:
                if meta.django_get_or_create:
                    node.boundary = self._new_boundary(
                        'atomic',
                        'get_or_create(%s) on database %r' % (
                            ', '.join(meta.django_get_or_create), alias),
                        commits=True,
                    )
                else:
                    node.boundary = self._new_boundary(
                        'autocommit',
                        'manager.create() INSERT on database %r' % alias,
                        commits=True,
                    )
            elif kind == StepKind.FINALIZE:
                node.boundary = self._new_boundary(
                    'autocommit',
                    'instance.save() UPDATE after post-generation on %r' % alias,
                    commits=True,
                )
            else:
                node.boundary = self._new_boundary(
                    'autocommit',
                    'statements of %r on database %r (Django autocommit)' % (
                        method_name or 'post-generation hook', alias),
                    commits=True,
                )
            node.boundary.steps.append(node)
            return

        node.boundary = None

    def _sa_shared(self, kind, label):
        if self._sa_shared_boundary is None:
            self._sa_shared_boundary = self._new_boundary(kind, label)
        return self._sa_shared_boundary


# ---------------------------------------------------------------------------
# Planning builders
# ---------------------------------------------------------------------------


class _PlanningStepBuilder(builder.StepBuilder):
    """A StepBuilder that records persistence steps instead of running them."""

    def __init__(self, factory_meta, extras, strategy, plan_ctx):
        super().__init__(factory_meta, extras, strategy)
        self.plan_ctx = plan_ctx

    def recurse(self, factory_meta, extras):
        return self.__class__(factory_meta, extras, self.strategy, self.plan_ctx)

    def build(self, parent_step=None, force_sequence=None):
        meta = self.factory_meta
        factory = meta.factory

        # Non-persistent factories (Dict/List/Stub) are evaluated for real:
        # they only build plain containers, never touch a database, and their
        # values (dict/list/stub) are consumed by surrounding declarations.
        if (meta.strategy == enums.STUB_STRATEGY
                or issubclass(factory, (base.DictFactory, base.ListFactory))):
            return self._build_computed(parent_step, force_sequence)

        ctx = self.plan_ctx
        ctx.note_meta(meta)

        pre, post = builder.parse_declarations(
            self.extras,
            base_pre=meta.pre_declarations,
            base_post=meta.post_declarations,
        )

        sequence = ctx.next_sequence(
            meta,
            force_sequence=force_sequence,
            force_init_sequence=self.force_init_sequence,
        )

        node = PlannedStep(
            ctx.next_step_id(),
            StepKind.INSTANTIATE,
            name='create %s' % factory.__name__,
            factory=factory,
            model=meta.get_model_class(),
        )
        ctx.attach(node, StepKind.INSTANTIATE, factory_name=factory.__name__)
        node.muted_signals = _muted_signals(factory._create)
        # Raw params are required by Django/SQLAlchemy get_or_create recovery.
        node.original_params = dict(self.extras)

        step = builder.BuildStep(self, sequence, parent_step)

        # Resolve pre-declarations: SubFactory chains attach as pre-children.
        previous_sink = ctx._pre_sink
        ctx._pre_sink = node
        try:
            step.resolve(pre)
        finally:
            ctx._pre_sink = previous_sink

        node.attributes = dict(step.attributes)
        args, kwargs = meta.prepare_arguments(step.attributes)
        node.args = args
        node.kwargs = kwargs

        placeholder = PlannedInstance(node)

        # Post-generation declarations; RelatedFactory recursion must not be
        # attached as a pre-child (sink already restored above).
        for declaration_name in post.sorted():
            declaration_with_context = post[declaration_name]
            post_node = _plan_post(
                ctx=ctx,
                factory=factory,
                step=step,
                placeholder=placeholder,
                name=declaration_name,
                declaration=declaration_with_context.declaration,
                overrides=declaration_with_context.context,
            )
            node.post_children.append(post_node)

        finalize = PlannedStep(
            ctx.next_step_id(),
            StepKind.FINALIZE,
            name='_after_postgeneration',
            factory=factory,
            model=node.model,
        )
        # Django re-saves only when at least one post hook ran and the
        # deprecation escape hatch isn't set; anything else is a no-op.
        will_resave = (
            bool(post.declarations)
            and not getattr(meta, 'skip_postgeneration_save', False)
            and hasattr(meta, 'django_get_or_create')
        )
        if will_resave:
            ctx.attach(finalize, StepKind.FINALIZE)
            finalize.muted_signals = _muted_signals(factory._after_postgeneration)
        node.post_children.append(finalize)

        if parent_step is None:
            ctx.roots.append(node)
        elif ctx._pre_sink is not None:
            ctx._pre_sink.pre_children.append(node)

        return placeholder

    def _build_computed(self, parent_step, force_sequence):
        """Evaluate a non-persistent factory normally (containers/stubs)."""
        extras = dict(self.extras)
        if self.force_init_sequence is not None:
            extras.setdefault('__sequence', self.force_init_sequence)
        # Plain builder, but recursion keeps planning so nested SubFactories
        # are recorded instead of being persisted.
        real_builder = _RoutedStepBuilder(
            self.factory_meta, extras, self.strategy,
            self.plan_ctx, self.__class__)
        return real_builder.build(
            parent_step=parent_step, force_sequence=force_sequence)


class _RoutedStepBuilder(builder.StepBuilder):
    """Plain builder that recurses through planning builders.

    Used for the *computed* evaluation of Dict/List/Stub factories: the
    container itself is built, but any SubFactory nested inside is planned.
    """

    def __init__(self, factory_meta, extras, strategy, plan_ctx, planning_cls):
        super().__init__(factory_meta, extras, strategy)
        self.plan_ctx = plan_ctx
        self.planning_cls = planning_cls

    def recurse(self, factory_meta, extras):
        return self.planning_cls(factory_meta, extras, self.strategy, self.plan_ctx)


# ---------------------------------------------------------------------------
# Post-generation planning
# ---------------------------------------------------------------------------


def _plan_post(*, ctx, factory, step, placeholder, name,
               declaration, overrides):
    """Plan a single post-generation declaration; returns a post node."""

    # Maybe: choose the branch eagerly whenever the decider is a pre-phase
    # declaration (the common case, e.g. Trait).
    if isinstance(declaration, declarations.Maybe):
        node = PlannedStep(
            ctx.next_step_id(), StepKind.POST_MAYBE,
            name='maybe %s' % name, declaration_name=name,
            factory=factory, model=step.builder.factory_meta.get_model_class())
        decider_phase = enums.get_builder_phase(declaration.decider)
        if decider_phase == enums.BuilderPhase.ATTRIBUTE_RESOLUTION:
            choice = declaration.decider.evaluate_pre(
                instance=step.stub, step=step, overrides={})
            target = declaration.yes if choice else declaration.no
            child = _plan_post(
                ctx=ctx, factory=factory, step=step, placeholder=placeholder,
                name=name, declaration=target, overrides=overrides)
            node.post_children.append(child)
            return node
        # A post-instantiation decider cannot be chosen without executing it.
        node.kind = StepKind.POST_GENERIC
        node.name = 'maybe(post-decider) %s' % name
        node.declaration = declaration
        node.overrides = overrides
        ctx.attach(node, StepKind.POST_HOOK, method_name=name)
        return node

    if isinstance(declaration, declarations.RelatedFactoryList):
        return _plan_related_list(
            ctx, factory, step, placeholder, name, declaration, overrides)

    if isinstance(declaration, declarations.RelatedFactory):
        return _plan_related(
            ctx, factory, step, placeholder, name, declaration, overrides)

    if isinstance(declaration, declarations.PostGenerationMethodCall):
        context = _capture_post_context(declaration, step, placeholder, overrides)
        node = PlannedStep(
            ctx.next_step_id(), StepKind.POST_METHOD,
            name='%s.%s' % (name, declaration.method_name),
            declaration_name=name, factory=factory,
            model=step.builder.factory_meta.get_model_class())
        node.method_name = declaration.method_name
        if context.value_provided:
            node.method_args = (context.value,)
        elif declaration.method_arg is not declarations.NotProvided:
            node.method_args = (declaration.method_arg,)
        else:
            node.method_args = ()
        node.method_kwargs = dict(context.extra)
        ctx.attach(node, StepKind.POST_METHOD,
                   method_name=declaration.method_name)
        return node

    if isinstance(declaration, declarations.PostGeneration):
        context = _capture_post_context(declaration, step, placeholder, overrides)
        node = PlannedStep(
            ctx.next_step_id(), StepKind.POST_HOOK,
            name=getattr(declaration.function, '__name__', 'post_generation'),
            declaration_name=name, factory=factory,
            model=step.builder.factory_meta.get_model_class())
        node.hook_function = declaration.function
        node.hook_context = context
        ctx.attach(node, StepKind.POST_HOOK, method_name=name)
        return node

    if declaration is declarations.SKIP or declaration is None:
        node = PlannedStep(
            ctx.next_step_id(), StepKind.POST_SKIPPED,
            name='skip %s' % name, declaration_name=name,
            factory=factory, model=step.builder.factory_meta.get_model_class())
        node.status = StepStatus.SKIPPED
        node.result = declarations.SKIP
        return node

    if not isinstance(declaration, declarations.PostGenerationDeclaration):
        node = PlannedStep(
            ctx.next_step_id(), StepKind.POST_VALUE,
            name='value %s' % name, declaration_name=name,
            factory=factory, model=step.builder.factory_meta.get_model_class())
        node.provided_value = declaration
        return node

    # Custom PostGenerationDeclaration: replay verbatim at execution time.
    node = PlannedStep(
        ctx.next_step_id(), StepKind.POST_GENERIC,
        name='custom %s' % name, declaration_name=name,
        factory=factory, model=step.builder.factory_meta.get_model_class())
    node.declaration = declaration
    node.overrides = dict(overrides)
    ctx.attach(node, StepKind.POST_HOOK, method_name=name)
    return node


def _capture_post_context(declaration, step, placeholder, overrides):
    """Resolve a post-generation context the same way evaluate_post would."""
    context = declaration.unroll_context(placeholder, step, overrides)
    return declarations.PostGenerationContext(
        value_provided=bool('' in context),
        value=context.get(''),
        extra={k: v for k, v in context.items() if k != ''},
    )


def _fill_related(ctx, post_node, step, placeholder, declaration, overrides):
    """Populate a RELATED node (shared by single & list variants)."""
    # RelatedFactory.unroll_context is a no-op; the raw context decides
    # whether the caller provided a value instead of generating one.
    if '' in overrides:
        post_node.related_child = None
        post_node.related_name = declaration.name
        post_node.provided_value = overrides.get('')
        post_node.result = post_node.provided_value
        post_node.status = StepStatus.SKIPPED
        return post_node

    related_factory = declaration.get_factory()
    passed_kwargs = dict(declaration.defaults)
    passed_kwargs.update({k: v for k, v in overrides.items() if k != ''})
    if declaration.name:
        passed_kwargs[declaration.name] = placeholder
    post_node.related_name = declaration.name

    child_placeholder = step.recurse(related_factory, passed_kwargs)
    post_node.related_child = child_placeholder.node
    post_node.boundary = child_placeholder.node.boundary
    return post_node


def _plan_related(ctx, factory, step, placeholder, name, declaration, overrides):
    post_node = PlannedStep(
        ctx.next_step_id(), StepKind.RELATED,
        name='related %s' % name, declaration_name=name,
        factory=factory, model=step.builder.factory_meta.get_model_class())
    return _fill_related(
        ctx, post_node, step, placeholder, declaration, overrides)


def _plan_related_list(ctx, factory, step, placeholder, name,
                       declaration, overrides):
    size = declaration.size
    count = size if isinstance(size, int) else size()
    group = PlannedStep(
        ctx.next_step_id(), StepKind.RELATED_LIST,
        name='related list %s' % name, declaration_name=name,
        factory=factory, model=step.builder.factory_meta.get_model_class())
    for _index in range(count):
        wrapper = PlannedStep(
            ctx.next_step_id(), StepKind.RELATED,
            name='related %s[%d]' % (name, _index), declaration_name=name,
            factory=factory, model=group.model)
        _fill_related(ctx, wrapper, step, placeholder, declaration, overrides)
        group.post_children.append(wrapper)
    return group


# ---------------------------------------------------------------------------
# mute_signals support
# ---------------------------------------------------------------------------


_MUTE_MARKER = '_factoryboy_muted_signals'


def _muted_signals(method):
    return getattr(method, _MUTE_MARKER, ()) or ()


def _signal_name(signal):
    return getattr(signal, 'name', None) or repr(signal)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


_EXECUTION_STRATEGY = types.SimpleNamespace(builder=types.SimpleNamespace(
    strategy=enums.CREATE_STRATEGY))


def _substitute(value, registry, plan):
    """Replace PlannedInstance placeholders with live instances."""
    if isinstance(value, PlannedInstance):
        node = value.node
        if node.id in registry:
            return registry[node.id]
        if node.status == StepStatus.DONE:
            return node.result
        # An embedded relation (e.g. a SubFactory inside a hook argument) that
        # is not part of the main tree: persist it on demand.
        return plan._execute_instantiate(node)
    if isinstance(value, dict):
        return {k: _substitute(v, registry, plan) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, registry, plan) for v in value]
    if isinstance(value, tuple):
        return tuple(_substitute(v, registry, plan) for v in value)
    if isinstance(value, set):
        return {_substitute(v, registry, plan) for v in value}
    return value


class PersistencePlan:
    """An executable, resumable description of a ``create()`` call.

    Returned by :meth:`factory.Factory.plan_create`.  Inspect :attr:`steps`
    (flat, ordered), :attr:`tree` and :attr:`transaction_boundaries` before
    calling :meth:`commit`.

    Calling :meth:`commit` repeatedly is safe: steps already marked done are
    skipped, so a retry after a failure never duplicates persisted objects.
    """

    def __init__(self, factory, root, ctx, *, kwargs):
        self.factory = factory
        self._root = root
        self._ctx = ctx
        self.kwargs = kwargs
        self._registry = {}
        self.error = None
        #: Signals muted for the whole plan by a class-level mute_signals.
        self.muted_signals = tuple(_muted_signals(factory._generate))
        self._execution_step = _EXECUTION_STRATEGY

    # -- introspection -----------------------------------------------------

    @property
    def tree(self):
        """Root :class:`PlannedStep` (the factory's instantiation node)."""
        return self._root

    @property
    def steps(self):
        """All steps in execution order, de-duplicated."""
        ordered = []
        seen = set()
        for step in self._root.iter_flat():
            if step.id not in seen:
                seen.add(step.id)
                ordered.append(step)
        return ordered

    @property
    def transaction_boundaries(self):
        """Ordered, de-duplicated transaction boundaries touched by the plan."""
        seen = set()
        boundaries = []
        for step in self.steps:
            boundary = step.boundary
            if boundary is not None and id(boundary) not in seen:
                seen.add(id(boundary))
                boundaries.append(boundary)
        return boundaries

    @property
    def completed_steps(self):
        """Steps whose effects are durable.

        Includes instantiation steps whose row was persisted even when a later
        post hook failed (and which therefore must not be run again).
        """
        return [
            s for s in self.steps
            if s.status == StepStatus.DONE or s.persisted
        ]

    @property
    def failed_step(self):
        for step in self.steps:
            if step.status == StepStatus.FAILED:
                return step
        return None

    @property
    def remaining_steps(self):
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def skipped_steps(self):
        return [s for s in self.steps if s.status == StepStatus.SKIPPED]

    @property
    def succeeded(self):
        return (
            self._root.status == StepStatus.DONE
            and self.failed_step is None
            and not self.remaining_steps
        )

    @property
    def instance(self):
        """The root instance once the plan committed, else None."""
        return self._root.result if self.succeeded else None

    def describe(self):
        """Return a human-readable rendering of the plan and its statuses."""
        lines = [
            'PersistencePlan for %s' % self.factory.__name__,
            self._root.describe(),
        ]
        boundaries = self.transaction_boundaries
        if boundaries:
            lines.append('Transaction boundaries:')
            for boundary in boundaries:
                lines.append('  - %s [%s, commits=%s]' % (
                    boundary.label, boundary.kind, boundary.commits))
        if self.muted_signals:
            lines.append('Muted signals (whole plan): %s' % ', '.join(
                _signal_name(s) for s in self.muted_signals))
        return '\n'.join(lines)

    def __repr__(self):
        return '<PersistencePlan for %s: %d step(s), %s>' % (
            self.factory.__name__, len(self.steps),
            'committed' if self.succeeded else 'pending')

    # -- execution ---------------------------------------------------------

    def commit(self):
        """Execute the plan, resuming from the first non-completed step.

        Returns the root model instance.  Raises :class:`PartialCommitError`
        (with the original error as ``__cause__``) if a step fails; the plan's
        per-step statuses are then updated and can be inspected.  Fix the
        issue and call ``commit()`` again to resume idempotently.
        """
        if self.succeeded:
            return self._root.result

        with self._mute_context():
            self._execute_instantiate(self._root)
        self.error = None
        return self._root.result

    @contextlib.contextmanager
    def _mute_context(self):
        if not self.muted_signals:
            yield
            return
        # Only importable when django is installed, which is guaranteed if a
        # mute marker exists at all.
        from .django import mute_signals
        with mute_signals(*self.muted_signals):
            yield

    def _fail(self, node, exc):
        node.status = StepStatus.FAILED
        node.error = exc
        self.error = exc
        raise PartialCommitError(self, node, exc) from exc

    def _execute_instantiate(self, node):
        if node.status == StepStatus.DONE and self._posts_complete(node):
            return node.result

        if not node.persisted:
            # Pre-relations (SubFactory chains), in planning order.
            for child in node.pre_children:
                self._execute_instantiate(child)

            args = _substitute(node.args, self._registry, self)
            kwargs = _substitute(node.kwargs, self._registry, self)

            meta = node.factory._meta
            # DjangoModelFactory / SQLAlchemyModelFactory rely on
            # _original_params for get_or_create recovery after IntegrityError.
            if hasattr(node.factory, '_original_params'):
                node.factory._original_params = node.original_params

            try:
                instance = meta.instantiate(
                    self._execution_step, args, kwargs)
            except Exception as exc:  # noqa: BLE001 - recorded on the step
                self._fail(node, exc)
            node.result = instance
            node.persisted = True
            self._registry[node.id] = instance

        node.status = StepStatus.DONE
        meta = node.factory._meta

        # Post-generation declarations, in the exact order of a normal build.
        # Already-completed steps are reused so a retry only reruns what was
        # not persisted.
        results = {}
        for post in node.post_children:
            if post.kind == StepKind.FINALIZE:
                if post.status != StepStatus.DONE:
                    try:
                        meta.use_postgeneration_results(
                            self._execution_step, node.result, results)
                    except Exception as exc:  # noqa: BLE001
                        self._fail(post, exc)
                    post.result = None
                    post.status = StepStatus.DONE
                continue

            if post.status == StepStatus.DONE:
                value = post.result
            elif post.status == StepStatus.SKIPPED:
                value = post.result
            else:
                try:
                    value = self._execute_post(post, node.result)
                except PartialCommitError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._fail(post, exc)
            if post.declaration_name is not None:
                results[post.declaration_name] = value

        return node.result

    def _posts_complete(self, node):
        """True if every post-generation subtree is done or intentionally skipped."""
        for post in node.post_children:
            for flat in _iter_post_flat(post):
                if flat.status not in (
                        StepStatus.DONE, StepStatus.SKIPPED):
                    return False
        return True

    def _execute_post(self, post, instance):
        kind = post.kind

        if kind == StepKind.POST_SKIPPED:
            post.status = StepStatus.SKIPPED
            return post.result

        if kind == StepKind.POST_VALUE:
            post.status = StepStatus.DONE
            return post.provided_value

        if kind == StepKind.POST_MAYBE:
            value = self._execute_post(post.post_children[0], instance)
            post.status = post.post_children[0].status
            return value

        if kind == StepKind.RELATED:
            if post.related_child is None:
                # Caller supplied a value (possibly None) for the relation.
                post.status = StepStatus.SKIPPED
                return _substitute(post.provided_value, self._registry, self)
            related = self._execute_instantiate(post.related_child)
            post.status = StepStatus.DONE
            post.result = related
            return related

        if kind == StepKind.RELATED_LIST:
            values = []
            for wrapper in post.post_children:
                values.append(self._execute_post(wrapper, instance))
            post.status = StepStatus.DONE
            post.result = values
            return values

        if kind == StepKind.POST_METHOD:
            args = _substitute(post.method_args, self._registry, self)
            kwargs = _substitute(post.method_kwargs, self._registry, self)
            value = getattr(instance, post.method_name)(*args, **kwargs)
            post.status = StepStatus.DONE
            post.result = value
            return value

        if kind == StepKind.POST_HOOK:
            context = post.hook_context
            value = _substitute(context.value, self._registry, self)
            extra = _substitute(context.extra, self._registry, self)
            result = post.hook_function(instance, True, value, **extra)
            post.status = StepStatus.DONE
            post.result = result
            return result

        if kind == StepKind.POST_GENERIC:
            # Build a live step so custom declarations can recurse normally.
            live_builder = builder.StepBuilder(
                post.factory._meta, {}, enums.CREATE_STRATEGY)
            live_step = builder.BuildStep(
                live_builder,
                post.factory._meta.next_sequence(),
                parent_step=None)
            overrides = _substitute(post.overrides, self._registry, self)
            result = post.declaration.evaluate_post(
                instance=instance, step=live_step, overrides=overrides)
            post.status = StepStatus.DONE
            post.result = result
            return result

        raise errors.FactoryError('Unknown planned step kind %r' % kind)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def plan_persistence(factory, call_kwargs):
    """Build a :class:`PersistencePlan` for ``factory.create(**call_kwargs)``.

    No framework persistence code runs; see the module docstring.
    """
    if factory._meta.abstract:
        raise errors.FactoryError(
            "Cannot plan instances of abstract factory %(f)s; "
            "Ensure %(f)s.Meta.model is set and %(f)s.Meta.abstract "
            "is either not set or False." % dict(f=factory.__name__))
    if factory._meta.strategy == enums.STUB_STRATEGY:
        raise errors.UnsupportedStrategy(
            'Cannot plan persistence for stub factory %r.' % factory.__name__)

    ctx = _PlanContext(factory)
    try:
        step_builder = _PlanningStepBuilder(
            factory._meta, dict(call_kwargs), enums.CREATE_STRATEGY, ctx)
        step_builder.build()
    finally:
        ctx.reset_iterators()

    if len(ctx.roots) != 1:  # pragma: no cover - defensive
        raise errors.FactoryError(
            'Unexpected plan shape: %d roots.' % len(ctx.roots))
    return PersistencePlan(
        factory, ctx.roots[0], ctx, kwargs=dict(call_kwargs))

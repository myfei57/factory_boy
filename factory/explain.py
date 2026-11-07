# Copyright: See the LICENSE file.


"""Static explanation of declaration override conflicts.

This module computes, for a given factory, which source wins for every
declared field, without ever evaluating a declaration, generating a factory
object, or exposing concrete values.

The precedence replayed here is the one implemented by
:class:`factory.FactoryOptions` and :func:`factory.builder.parse_declarations`:

1. declarations inherited from parent factories (MRO order);
2. declarations defined directly on the factory;
3. simple ``class Params`` entries;
4. :class:`~factory.Trait` parameters (conditional overrides, wrapped in a
   :class:`~factory.Maybe`), in parameter creation order;
5. keyword arguments passed at call time (``MyFactory.build(...)``);
6. for :class:`~factory.SubFactory` fields, the nested kwargs win over the
   sub-factory defaults, which themselves win over the nested factory's own
   declarations.

The report intentionally records *categories* of values (declaration class or
plain Python type name) rather than the values themselves.
"""

import ast
import collections
import inspect
import textwrap

from . import base, builder, declarations, enums, utils


# Sources (value categories) recorded in override chains.
class OverrideSource:
    #: Declaration inherited from a factory further up the MRO.
    INHERITED = 'inherited_declaration'
    #: Declaration defined directly on the analyzed factory.
    BASE = 'base_declaration'
    #: Simple entry of a ``class Params`` section.
    PARAM = 'param'
    #: Conditional override injected by a :class:`~factory.Trait`.
    TRAIT = 'trait'
    #: Keyword argument passed at call time (build/create/stub).
    KWARG = 'call_kwarg'
    #: Default declared inside a ``SubFactory(OtherFactory, field=...)``.
    SUBFACTORY_DEFAULT = 'subfactory_default'
    #: Context pushed into a sub-factory through ``outer__inner=...`` kwargs.
    SUBFACTORY_CONTEXT = 'subfactory_context'
    #: Value handed to a post-generation declaration or a Transformer.
    POSTGENERATION_VALUE = 'post_generation_value'


# Kinds of cyclic risks detected statically.
class CycleRiskKind:
    #: Sub-factory graph loops back onto a factory already being built.
    SUBFACTORY_RECURSION = 'subfactory_recursion'
    #: Params/Trait parameters depend on each other.
    PARAMETER_DEPENDENCY = 'parameter_dependency'
    #: Lazy/Self attributes reference each other in a cycle.
    LAZY_ATTRIBUTE = 'lazy_attribute'


def describe_value(value):
    """Return a value-free category descriptor for a declaration/value."""
    if value is declarations.SKIP:
        return 'Skip'
    phase = enums.get_builder_phase(value)
    if phase is not None:
        # Declaration instance: expose its structural type only.
        return type(value).__name__
    return type(value).__name__


class OverrideStep:
    """One layer of a field's override chain.

    Attributes:
        source: one of :class:`OverrideSource`, the category of this writer.
        origin: structural location of the writer (factory / Params / trait
            names). Never contains field values.
        kind: value-free descriptor of what this layer holds (declaration
            class name or plain Python type name).
        phase: builder phase ('attributes' / 'post_instance') or None.
        conditional_on: when this layer only wins if a trait flag is enabled,
            name of that trait flag.
        branch: 'yes' for the active branch of a conditional layer, else None.
    """

    __slots__ = ('source', 'origin', 'kind', 'phase', 'conditional_on', 'branch')

    def __init__(self, source, origin, kind, phase=None, conditional_on=None,
                 branch=None):
        self.source = source
        self.origin = origin
        self.kind = kind
        self.phase = phase
        self.conditional_on = conditional_on
        self.branch = branch

    def as_dict(self):
        return {
            'source': self.source,
            'origin': self.origin,
            'kind': self.kind,
            'phase': self.phase,
            'conditional_on': self.conditional_on,
            'branch': self.branch,
        }

    def __repr__(self):
        return '<OverrideStep %s from %r (%s)>' % (
            self.source, self.origin, self.kind)


class AbandonedValue:
    """A layer that lost the override conflict.

    Attributes:
        source/origin/kind/conditional_on: description of the abandoned layer.
        reason: why the layer is abandoned. 'replaced' for simple overrides,
            'excluded' for Meta.exclude, 'hidden_param' for Params-only flags
            stripped before instantiation, 'shadowed_by_rename' for fields
            overwritten by a Meta.rename hop, 'skip' when the declaration
            resolves to SKIP.
    """

    __slots__ = ('source', 'origin', 'kind', 'reason', 'conditional_on')

    def __init__(self, step, reason, conditional_on=None):
        self.source = step.source
        self.origin = step.origin
        self.kind = step.kind
        self.reason = reason
        self.conditional_on = conditional_on

    def as_dict(self):
        return {
            'source': self.source,
            'origin': self.origin,
            'kind': self.kind,
            'reason': self.reason,
            'conditional_on': self.conditional_on,
        }

    def __repr__(self):
        return '<AbandonedValue %s from %r (%s)>' % (
            self.source, self.origin, self.reason)


class CycleRisk:
    """A cyclic risk detected from declarations alone.

    Attributes:
        kind: one of :class:`CycleRiskKind`.
        path: ordered list of factory/field names closing the cycle.
        fields: field names involved in the cycle.
    """

    __slots__ = ('kind', 'path', 'fields')

    def __init__(self, kind, path, fields):
        self.kind = kind
        self.path = list(path)
        self.fields = sorted(fields)

    def as_dict(self):
        return {
            'kind': self.kind,
            'path': list(self.path),
            'fields': list(self.fields),
        }

    def __repr__(self):
        return '<CycleRisk %s %s>' % (self.kind, ' -> '.join(self.path))


class FieldOverride:
    """Override explanation for a single field."""

    __slots__ = (
        'field', 'phase', 'chain', 'abandoned', 'context_chain',
        'excluded', 'inline_arg', 'parameter_flag', 'renamed_to',
        'renamed_from', 'shadowed_by_rename', 'delivered', 'subfactory',
        'cycle_risks',
    )

    def __init__(self, field):
        self.field = field
        self.phase = None
        #: Override layers, from weakest to strongest.
        self.chain = []
        #: Layers dropped along the way.
        self.abandoned = []
        #: Non-replacing context writes (subpath, step), e.g. postgen values.
        self.context_chain = []
        self.excluded = False
        self.inline_arg = None
        self.parameter_flag = False
        self.renamed_to = None
        self.renamed_from = None
        self.shadowed_by_rename = None
        self.delivered = True
        #: Nested OverrideReport when the winner is a SubFactory.
        self.subfactory = None
        self.cycle_risks = []

    @property
    def winner(self):
        """The final structural source for the field, or None."""
        return self.chain[-1] if self.chain else None

    @property
    def has_conflict(self):
        return bool(self.abandoned) or len(self.chain) > 1

    def as_dict(self):
        return {
            'field': self.field,
            'phase': self.phase,
            'winner': self.winner.as_dict() if self.winner else None,
            'chain': [step.as_dict() for step in self.chain],
            'abandoned': [value.as_dict() for value in self.abandoned],
            'context_chain': [
                {'subpath': subpath, 'step': step.as_dict()}
                for subpath, step in self.context_chain
            ],
            'excluded': self.excluded,
            'inline_arg': self.inline_arg,
            'parameter_flag': self.parameter_flag,
            'renamed_to': self.renamed_to,
            'renamed_from': self.renamed_from,
            'shadowed_by_rename': self.shadowed_by_rename,
            'delivered': self.delivered,
            'subfactory': self.subfactory.as_dict() if self.subfactory else None,
            'cycle_risks': [risk.as_dict() for risk in self.cycle_risks],
        }


class OverrideReport:
    """Result of :func:`explain_overrides`.

    Iterating the report yields field names in stable (alphabetical) order.
    """

    def __init__(self, factory):
        self.factory = factory
        self._fields = collections.OrderedDict()
        #: Structural problems factory_boy would raise at call time.
        self.errors = []
        #: Trait parameters declared without any overridden field.
        self.empty_traits = []

    def _add(self, field):
        if field not in self._fields:
            self._fields[field] = FieldOverride(field)
        return self._fields[field]

    def sorted_field_names(self):
        return sorted(self._fields)

    def __iter__(self):
        return iter(self.sorted_field_names())

    def __contains__(self, field):
        return field in self._fields

    def __getitem__(self, field):
        return self._fields[field]

    def get(self, field, default=None):
        return self._fields.get(field, default)

    @property
    def conflicts(self):
        """Field names with more than one competing source, sorted."""
        return [
            name for name in self.sorted_field_names()
            if self._fields[name].has_conflict
        ]

    @property
    def cycle_risks(self):
        risks = []
        seen = set()
        for name in self.sorted_field_names():
            for risk in self._fields[name].cycle_risks:
                key = (risk.kind, tuple(risk.path), tuple(risk.fields))
                if key not in seen:
                    seen.add(key)
                    risks.append(risk)
        return risks

    def as_dict(self):
        """Return a JSON-friendly description of the whole report."""
        return {
            'factory': '%s.%s' % (
                self.factory.__module__, self.factory.__name__),
            'fields': collections.OrderedDict(
                (name, self._fields[name].as_dict())
                for name in self.sorted_field_names()
            ),
            'conflicts': self.conflicts,
            'empty_traits': list(self.empty_traits),
            'cycle_risks': [risk.as_dict() for risk in self.cycle_risks],
            'errors': list(self.errors),
        }


class _InputLayer:
    """An annotated write entering a factory level (never evaluated)."""

    __slots__ = ('value', 'source', 'origin', 'conditional_on')

    def __init__(self, value, source, origin, conditional_on=None):
        self.value = value
        self.source = source
        self.origin = origin
        self.conditional_on = conditional_on


def _param_origins(meta):
    """Map each parameter name to the factory that most recently defined it.

    Own ``class Params`` entries replace inherited ones; identity comparison
    mirrors the metaclass' parent-then-own parameters update.
    """
    origins = {}
    for name, value in meta.parameters.items():
        owner = meta.factory.__name__
        for parent in reversed(meta.factory.__mro__[1:]):
            parent_meta = getattr(parent, '_meta', None)
            if (parent_meta is not None
                    and name in parent_meta.parameters
                    and parent_meta.parameters[name] is value):
                owner = parent.__name__
        origins[name] = owner
    return origins


def _declaration_origins(meta):
    """Yield (name, value, origin_factory) base-declaration writes in MRO order."""
    seen = {}
    for parent in reversed(meta.factory.__mro__[1:]):
        parent_meta = getattr(parent, '_meta', None)
        if parent_meta is None:
            continue
        for name, value in parent_meta.base_declarations.items():
            if seen.get(name) is not value:
                seen[name] = value
                yield name, value, parent.__name__
    for name, value in vars(meta.factory).items():
        if meta._is_declaration(name, value) and seen.get(name) is not value:
            seen[name] = value
            yield name, value, meta.factory.__name__


def _iter_declaration_references(value):
    """Yield same-level field names statically referenced by a declaration."""
    seen = set()

    def walk(node):
        if id(node) in seen:
            return
        seen.add(id(node))

        if isinstance(node, declarations.SelfAttribute):
            if node.depth <= 1:
                yield node.attribute_name.split('.', 1)[0]
        elif isinstance(node, declarations.Maybe):
            yield from walk(node.decider)
            yield from walk(node.yes)
            yield from walk(node.no)
        elif isinstance(node, declarations.Transformer):
            yield from walk(node.default)
        elif isinstance(node, declarations.SubFactory):
            # Sub-factory defaults resolve against the nested factory's
            # stub, never the current one; they are analyzed when recursing.
            return
        elif isinstance(node, declarations.BaseDeclaration):
            for context_value in getattr(node, '_defaults', {}).values():
                yield from walk(context_value)

    yield from walk(value)


def _iter_subfactory_leaves(value):
    """Yield SubFactory declarations hidden inside wrapper declarations."""
    if isinstance(value, declarations.SubFactory):
        yield value
    elif isinstance(value, declarations.Maybe):
        yield from _iter_subfactory_leaves(value.yes)
        yield from _iter_subfactory_leaves(value.no)
    elif isinstance(value, declarations.Transformer):
        yield from _iter_subfactory_leaves(value.default)


def _static_function_references(function):
    """Best-effort extraction of stub attribute names read by a function.

    Only the first argument (the lazy stub) is inspected, through ``ast``;
    the function itself is never called.
    """
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return ()
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return ()

    refs = set()
    functional_nodes = (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, functional_nodes) or not node.args.args:
            continue
        arg_name = node.args.args[0].arg
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Attribute):
                continue
            current = sub
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name) and current.id == arg_name:
                target = sub
                while isinstance(target.value, ast.Attribute):
                    target = target.value
                refs.add(target.attr)
    return refs


def _inverse_rename(rename):
    return {new: old for old, new in rename.items()}


def _resolve_factory(declaration):
    """Retrieve a sub-factory class without generating anything.

    Importing a lazily-referenced factory module does not execute the
    factory; failures are reported rather than raised.
    """
    try:
        return declaration.get_factory()
    except Exception:  # noqa: BLE001 - analysis must never crash on bad paths
        return None


def _find_cycles(edges):
    """Return {node: cycle_path} for nodes belonging to a directed cycle."""
    cycles = {}
    state = {}  # None/unset = unseen, 0 = on stack, 1 = finished
    stack = []

    def visit(node):
        state[node] = 0
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            if state.get(nxt) == 0:
                start = stack.index(nxt)
                path = stack[start:] + [nxt]
                for member in path[:-1]:
                    cycles.setdefault(member, path)
            elif state.get(nxt) is None:
                visit(nxt)
        stack.pop()
        state[node] = 1

    nodes = set(edges)
    for nexts in edges.values():
        nodes.update(nexts)
    for node in sorted(nodes):
        if state.get(node) is None:
            visit(node)
    return cycles


class _Analyzer:
    def analyze(self, factory, extra_layers=None, stack=None, errors=None,
                empty_traits=None):
        """Build the OverrideReport for one factory level.

        Args:
            factory: a Factory subclass.
            extra_layers: dict mapping a (possibly nested) kwarg name to the
                ordered list of _InputLayer writes for it (later wins).
            stack: tuple of factory classes currently being analyzed, for
                sub-factory cycle detection.
            errors/empty_traits: shared accumulators across nested reports.
        """
        if not isinstance(factory, type) or not issubclass(
                factory, base.BaseFactory):
            raise TypeError('explain_overrides expects a Factory subclass')

        meta = factory._meta
        stack = stack if stack is not None else (factory,)
        errors = errors if errors is not None else []
        empty_traits = empty_traits if empty_traits is not None else []
        report = OverrideReport(factory)
        report.errors = errors
        report.empty_traits = empty_traits

        # root field -> ordered winning steps
        chains = collections.defaultdict(list)
        # root -> subpath -> [(step, structural value)]
        contexts = collections.defaultdict(
            lambda: collections.defaultdict(list))
        # flat structural mirror of FactoryOptions.declarations
        values = {}

        def record_write(key, layer):
            root, sub = builder.DeclarationSet.split(key)
            step = OverrideStep(
                source=layer.source,
                origin=layer.origin,
                kind=describe_value(layer.value),
                phase=enums.get_builder_phase(layer.value),
                conditional_on=layer.conditional_on,
                branch='yes' if layer.conditional_on else None,
            )
            if sub is None:
                bucket = chains[root]
                if not bucket or values.get(root) is not layer.value:
                    bucket.append(step)
                values[root] = layer.value
            else:
                contexts[root][sub].append((step, layer.value))
                values[builder.DeclarationSet.join(root, sub)] = layer.value

        origin_name = factory.__name__

        # Layers 1 & 2: inherited then direct declarations.
        for name, value, source_factory in _declaration_origins(meta):
            source = (
                OverrideSource.BASE if source_factory == origin_name
                else OverrideSource.INHERITED)
            record_write(name, _InputLayer(
                value=value, source=source, origin=source_factory))

        # Layers 3 & 4: class Params (simple params + traits), creation order.
        param_origins = _param_origins(meta)
        trait_decider_edges = []
        ordered_parameters = utils.sort_ordered_objects(
            meta.parameters.items(), getter=lambda item: item[1])
        for name, parameter in ordered_parameters:
            source_factory = param_origins.get(name, origin_name)
            if isinstance(parameter, declarations.Trait):
                origin = '%s.Params.%s' % (source_factory, name)
                if not parameter.overrides:
                    record = origin
                    if record not in empty_traits:
                        empty_traits.append(record)
                for maybe_field in parameter.overrides:
                    if enums.SPLITTER not in maybe_field:
                        root, _ = builder.DeclarationSet.split(maybe_field)
                        trait_decider_edges.append((root, name))
                source = OverrideSource.TRAIT
            else:
                origin = '%s.Params.%s' % (source_factory, name)
                source = OverrideSource.PARAM
            for target, value in parameter.as_declarations(
                    name, dict(values)).items():
                record_write(target, _InputLayer(
                    value=value,
                    source=source,
                    origin=origin,
                    conditional_on=(
                        name if source == OverrideSource.TRAIT else None),
                ))

        # Layers 5 & 6: call-time kwargs / parent-provided sub-factory extras.
        extra_layers = extra_layers or {}
        post_roots = set()
        for key, layers in extra_layers.items():
            if key == '__sequence':
                # Consumed by StepBuilder before declarations are parsed.
                continue
            root, sub = builder.DeclarationSet.split(key)
            for layer in layers:
                value = layer.value
                phase = enums.get_builder_phase(value)
                is_post_root = bool(chains.get(root)) and (
                    root in post_roots
                    or chains[root][-1].phase
                    == enums.BuilderPhase.POST_INSTANTIATION)
                if sub is None:
                    if phase == enums.BuilderPhase.POST_INSTANTIATION:
                        if root in chains and not is_post_root:
                            # parse_declarations raises InvalidDeclarationError.
                            errors.append(
                                '%s: post-instantiation override for %r '
                                'shadows a base declaration'
                                % (origin_name, root))
                            continue
                        record_write(root, layer)
                        post_roots.add(root)
                    elif is_post_root:
                        # Flat value for a post-generation hook: becomes ''.
                        contexts[root][''].append((OverrideStep(
                            source=OverrideSource.POSTGENERATION_VALUE,
                            origin=layer.origin,
                            kind=describe_value(value),
                            phase=enums.BuilderPhase.POST_INSTANTIATION,
                        ), value))
                    elif (root in values
                          and isinstance(values[root], declarations.BaseDeclaration)
                          and values[root].CAPTURE_OVERRIDES
                          and enums.get_builder_phase(values[root])
                          == enums.BuilderPhase.ATTRIBUTE_RESOLUTION):
                        # Transformer: override captured under the '' subpath.
                        contexts[root][''].append((OverrideStep(
                            source=layer.source,
                            origin=layer.origin,
                            kind=describe_value(value),
                        ), value))
                    else:
                        record_write(root, layer)
                else:
                    if root not in values and not chains.get(root):
                        # parse_declarations raises InvalidDeclarationError.
                        errors.append(
                            '%s: deep override %r targets an undeclared '
                            'field' % (
                                origin_name,
                                builder.DeclarationSet.join(root, sub)))
                        continue
                    contexts[root][sub].append((OverrideStep(
                        source=layer.source,
                        origin=layer.origin,
                        kind=describe_value(value),
                        conditional_on=layer.conditional_on,
                        branch='yes' if layer.conditional_on else None,
                    ), value))

        # Field universe: all roots, params, exclude/rename entries.
        field_names = set(chains) | set(contexts) | set(meta.parameters)
        field_names.update(meta.exclude)
        field_names.update(meta.rename)
        field_names.update(meta.rename.values())
        field_names.discard('')
        for name in field_names:
            report._add(name)

        # Reference graph for lazy-attribute cycle detection.
        edges = collections.defaultdict(set)
        for root, name in trait_decider_edges:
            edges[root].add(name)
        for root, chain in chains.items():
            if not chain:
                continue
            if chain[-1].phase == enums.BuilderPhase.POST_INSTANTIATION:
                continue
            value = values.get(root)
            if value is None:
                continue
            for ref in _iter_declaration_references(value):
                edges[root].add(ref)
            if isinstance(value, (declarations.LazyAttribute,
                                  declarations.LazyAttributeSequence,
                                  declarations.ContainerAttribute)):
                for ref in _static_function_references(value.function):
                    edges[root].add(ref)
        lazy_cycles = _find_cycles(edges)
        param_cycles = self._parameter_cycles(meta)

        inline_positions = {
            name: pos for pos, name in enumerate(meta.inline_args)}

        for name in report.sorted_field_names():
            field = report[name]
            field.chain = list(chains.get(name, ()))
            field.phase = field.chain[-1].phase if field.chain else None
            if name in post_roots:
                field.phase = enums.BuilderPhase.POST_INSTANTIATION

            # Abandoned layers from pairwise override relationships.
            for previous, current in zip(field.chain, field.chain[1:]):
                field.abandoned.append(AbandonedValue(
                    previous,
                    reason='replaced',
                    conditional_on=current.conditional_on,
                ))

            for subpath in sorted(contexts.get(name, {})):
                for step, _ in contexts[name][subpath]:
                    field.context_chain.append((subpath, step))

            if name in meta.parameters:
                field.parameter_flag = True

            if name in meta.exclude:
                field.excluded = True
                if field.winner is not None:
                    field.abandoned.append(AbandonedValue(
                        field.winner, reason='excluded'))
                field.delivered = False
            elif field.parameter_flag:
                # Params-only fields are stripped in prepare_arguments.
                field.delivered = False
                if field.winner is not None:
                    field.abandoned.append(AbandonedValue(
                        field.winner, reason='hidden_param'))

            if name in meta.rename:
                field.renamed_to = meta.rename[name]
            inverse_rename = _inverse_rename(meta.rename)
            if name in inverse_rename:
                field.renamed_from = inverse_rename[name]

            if name in inline_positions:
                field.inline_arg = inline_positions[name]

            if values.get(name) is declarations.SKIP:
                field.abandoned.append(AbandonedValue(
                    field.winner, reason='skip'))
                field.delivered = False

            if field.phase == enums.BuilderPhase.POST_INSTANTIATION:
                # Post-generation hooks never reach the model constructor.
                field.delivered = False

            if name in lazy_cycles:
                field.cycle_risks.append(CycleRisk(
                    CycleRiskKind.LAZY_ATTRIBUTE,
                    lazy_cycles[name], set(lazy_cycles[name])))
            if name in param_cycles:
                field.cycle_risks.append(CycleRisk(
                    CycleRiskKind.PARAMETER_DEPENDENCY,
                    param_cycles[name], set(param_cycles[name])))

        # Meta.rename shadowing: kwargs[new] = kwargs.pop(old) overwrites new.
        for old_name, new_name in meta.rename.items():
            old_field = report.get(old_name)
            new_field = report.get(new_name)
            if (old_field is not None and old_field.delivered
                    and new_field is not None
                    and new_field.winner is not None
                    and new_field.delivered):
                new_field.shadowed_by_rename = old_name
                new_field.abandoned.append(AbandonedValue(
                    new_field.winner, reason='shadowed_by_rename'))

        # Static SubFactory recursion (no evaluation, no generation).
        for name in report.sorted_field_names():
            field = report[name]
            winning = values.get(name)
            if winning is None:
                continue
            if (field.winner is not None
                    and field.winner.source == OverrideSource.KWARG):
                # A call-time kwarg replaces the declaration entirely.
                continue
            leaves = list(_iter_subfactory_leaves(winning))
            for index, leaf in enumerate(leaves):
                nested, risk = self._recurse_subfactory(
                    field_name=name,
                    declaration=leaf,
                    factory=factory,
                    context_writes=contexts.get(name, {}),
                    stack=stack,
                    errors=errors,
                    empty_traits=empty_traits,
                )
                # The first (active-branch for Maybe) leaf is the primary one.
                if index == 0:
                    field.subfactory = nested
                if risk is not None:
                    field.cycle_risks.append(risk)

        return report

    def _recurse_subfactory(self, *, field_name, declaration, factory,
                            context_writes, stack, errors, empty_traits):
        target = _resolve_factory(declaration)

        merged = collections.defaultdict(list)
        # SubFactory defaults first, outer context second (context wins),
        # mirroring BaseDeclaration.unroll_context.
        for key, value in declaration._defaults.items():
            merged[key].append(_InputLayer(
                value=value,
                source=OverrideSource.SUBFACTORY_DEFAULT,
                origin='%s.%s(default)' % (factory.__name__, field_name),
            ))
        for subpath in sorted(context_writes):
            if subpath == '':
                continue
            for step, value in context_writes[subpath]:
                # Trait-provided deep contexts keep their conditional nature;
                # plain call-time kwargs become sub-factory contexts.
                source = step.source
                if source == OverrideSource.KWARG:
                    source = OverrideSource.SUBFACTORY_CONTEXT
                merged[subpath].append(_InputLayer(
                    value=value,
                    source=source,
                    origin=step.origin,
                    conditional_on=step.conditional_on,
                ))

        if target is None:
            errors.append(
                '%s: nested factory for %r could not be resolved without '
                'executing code' % (factory.__name__, field_name))
            placeholder = OverrideReport(factory)
            placeholder._add(field_name)
            placeholder.errors = errors
            placeholder.empty_traits = empty_traits
            return placeholder, None

        if target in stack:
            path = [f.__name__ for f in stack] + [
                '%s via %s' % (target.__name__, field_name)]
            risk = CycleRisk(
                CycleRiskKind.SUBFACTORY_RECURSION,
                path, {field_name})
            placeholder = OverrideReport(target)
            placeholder._add(field_name)
            placeholder.errors = errors
            placeholder.empty_traits = empty_traits
            placeholder[field_name].cycle_risks.append(risk)
            return placeholder, risk

        nested = self.analyze(
            target, dict(merged), stack + (target,),
            errors=errors, empty_traits=empty_traits)
        return nested, None

    def _parameter_cycles(self, meta):
        """Replicate FactoryOptions._check_parameter_dependencies statically."""
        deep_revdeps = collections.defaultdict(set)
        for name, parameter in meta.parameters.items():
            if not isinstance(parameter, declarations.Parameter):
                continue
            revdeps = parameter.get_revdeps(meta.parameters)
            if not revdeps:
                continue
            deep_revdeps[name] = set.union(
                *(deep_revdeps[dep] for dep in revdeps))
            deep_revdeps[name] |= set(revdeps)
        return {name: [name] for name, deps in deep_revdeps.items()
                if name in deps}


def explain_overrides(factory, **overrides):
    """Explain declaration override conflicts for ``factory``.

    Args:
        factory: a :class:`~factory.Factory` subclass to inspect.
        **overrides: the keyword arguments that would be passed to a
            build/create/stub call. Values are treated opaquely: they are
            never evaluated or rendered in the report.

    Returns:
        OverrideReport: a per-field report, with field names sorted
            alphabetically, describing each field's override chain, the
            categories of abandoned values, and any statically detectable
            cyclic risks.

    The function never calls a factory, a declaration evaluation hook, or a
    model constructor.
    """
    extra_layers = {
        key: [_InputLayer(
            value=value,
            source=OverrideSource.KWARG,
            origin='call(%s)' % factory.__name__,
        )]
        for key, value in overrides.items()
    }
    return _Analyzer().analyze(factory, extra_layers)

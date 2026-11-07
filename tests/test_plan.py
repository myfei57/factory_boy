# Copyright: See the LICENSE file.

"""Tests for the two-phase persistence plan (Factory.plan_create / commit)."""

import os
import unittest
import warnings

import factory
from factory import errors
from factory.plan import PartialCommitError, PersistencePlan, StepKind, StepStatus

try:
    import sqlalchemy
    from sqlalchemy import Column, ForeignKey, Integer, Unicode
    from sqlalchemy.orm import relationship
except ImportError:  # pragma: no cover
    sqlalchemy = None

try:
    import django
except ImportError:  # pragma: no cover
    django = None


# ---------------------------------------------------------------------------
# Plain-Python model stand-ins
# ---------------------------------------------------------------------------


class Model:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Pointed(Model):
    pass


class Pointer(Model):
    pass


#: list of (model_class, kwargs) recorded every time a generic factory creates.
creates = []


class RecorderFactory(factory.Factory):
    """Abstract factory whose _create calls are recorded, like a fake ORM."""

    class Meta:
        abstract = True

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        creates.append((model_class, kwargs))
        return model_class(*args, **kwargs)


class PointedFactory(RecorderFactory):
    class Meta:
        model = Pointed

    foo = factory.Sequence(lambda n: 'foo%d' % n)


class PointerFactory(RecorderFactory):
    class Meta:
        model = Pointer

    bar = 'bar'
    pointed = factory.SubFactory(PointedFactory)


class PointedRelatedFactory(PointedFactory):
    pointer = factory.RelatedFactory(
        PointerFactory, factory_related_name='pointed')


# ---------------------------------------------------------------------------
# Generic (ORM-agnostic) tests
# ---------------------------------------------------------------------------


class PlanStructureTests(unittest.TestCase):
    def setUp(self):
        creates.clear()
        PointedFactory.reset_sequence(0)

    def test_plan_does_not_invoke_create(self):
        plan = PointedRelatedFactory.plan_create()
        self.assertIsInstance(plan, PersistencePlan)
        self.assertEqual(creates, [])  # preview touches no "database"
        self.assertTrue(all(isinstance(s, factory.PlannedStep)
                            for s in plan.steps))

    def test_steps_are_ordered(self):
        plan = PointedRelatedFactory.plan_create()
        kinds = [s.kind for s in plan.steps]
        # Root instantiation first, then RelatedFactory wrapper + child,
        # finalize steps last (child's first).
        self.assertEqual(kinds, [
            StepKind.INSTANTIATE,          # Pointed (root)
            StepKind.RELATED,             # pointer wrapper
            StepKind.INSTANTIATE,         # Pointer
            StepKind.FINALIZE,            # Pointer finalize
            StepKind.FINALIZE,            # Pointed finalize
        ])
        root = plan.tree
        self.assertIs(root, plan.steps[0])
        self.assertEqual(root.pre_children, [])  # pointer is post, not pre
        related = root.post_children[0]
        self.assertEqual(related.kind, StepKind.RELATED)
        self.assertIs(related.related_child.factory, PointerFactory)

    def test_pre_relation_orders_before_parent(self):
        plan = PointerFactory.plan_create()
        kinds = [s.kind for s in plan.steps]
        # The subfactory is built completely (including its finalize step)
        # before the parent is instantiated.
        self.assertEqual(kinds, [
            StepKind.INSTANTIATE,   # pointed subfactory
            StepKind.FINALIZE,      # pointed post-generation finalize
            StepKind.INSTANTIATE,   # pointer root
            StepKind.FINALIZE,
        ])
        self.assertEqual(plan.tree.pre_children[0].model, Pointed)

    def test_resolved_kwargs_are_previewed(self):
        plan = PointedRelatedFactory.plan_create(foo='explicit')
        self.assertEqual(plan.tree.kwargs['foo'], 'explicit')

        plan = PointedRelatedFactory.plan_create()
        self.assertEqual(plan.tree.kwargs['foo'], 'foo0')

    def test_deep_related_context_is_previewed(self):
        plan = PointedRelatedFactory.plan_create(pointer__bar='deep')
        related = next(s for s in plan.steps if s.kind == StepKind.RELATED
                       and s.related_child is not None)
        self.assertEqual(related.related_child.kwargs['bar'], 'deep')
        # parent injected under the related name
        self.assertIsInstance(
            related.related_child.kwargs['pointed'],
            factory.plan.PlannedInstance)

    def test_plan_captures_same_values_as_create(self):
        plan = PointedRelatedFactory.plan_create()
        planned = plan.commit()
        self.assertEqual(creates[0][0], Pointed)
        self.assertEqual(creates[0][1]['foo'], planned.foo)
        self.assertEqual(creates[1][0], Pointer)
        self.assertEqual(creates[1][1]['bar'], 'bar')
        self.assertIs(creates[1][1]['pointed'], planned)

    def test_commit_twice_is_a_noop(self):
        plan = PointedRelatedFactory.plan_create()
        first = plan.commit()
        count = len(creates)
        second = plan.commit()
        self.assertIs(first, second)
        self.assertEqual(len(creates), count)

    def test_sequence_counter_is_untouched_by_planning(self):
        PointedFactory.reset_sequence(0)
        PointedRelatedFactory.plan_create()
        PointedRelatedFactory.plan_create()
        obj = PointedRelatedFactory.create()
        self.assertEqual(obj.foo, 'foo0')

    def test_iterator_is_rewound_after_planning(self):
        class F(PointedFactory):
            letter = factory.Iterator(['a', 'b', 'c'])

        plan = F.plan_create()
        self.assertEqual(plan.tree.kwargs['letter'], 'a')
        # A discarded plan must not consume iterator values.
        self.assertEqual(F.create().letter, 'a')
        self.assertEqual(F.create().letter, 'b')

    def test_lazy_attribute_reads_placeholder_relation(self):
        class F(PointerFactory):
            pointed_foo = factory.LazyAttribute(
                lambda o: 'saw-' + o.pointed.foo)

        plan = F.plan_create()
        self.assertEqual(plan.tree.kwargs['pointed_foo'], 'saw-foo0')
        ptr = plan.commit()
        self.assertEqual(ptr.pointed_foo, 'saw-foo0')

    def test_abstract_factory_rejected(self):
        class Abstract(factory.Factory):
            class Meta:
                abstract = True

        with self.assertRaises(errors.FactoryError):
            Abstract.plan_create()

    def test_stub_factory_rejected(self):
        with self.assertRaises(errors.UnsupportedStrategy):
            factory.StubFactory.plan_create(one=1)

    def test_describe_contains_steps_and_status(self):
        plan = PointedRelatedFactory.plan_create()
        text = plan.describe()
        self.assertIn('PersistencePlan', text)
        self.assertIn('instantiate+save Pointed', text)
        self.assertIn(StepStatus.PENDING, text)
        plan.commit()
        self.assertIn(StepStatus.DONE, plan.describe())


class PostGenerationPlanTests(unittest.TestCase):
    def setUp(self):
        creates.clear()
        PointedFactory.reset_sequence(0)

    def test_post_generation_method_call(self):
        calls = []

        class WithMethod(Model):
            def set_tag(self, tag):
                calls.append(tag)
                self.tag = tag

        class F(RecorderFactory):
            class Meta:
                model = WithMethod

            tag = factory.PostGenerationMethodCall('set_tag', 'default-tag')

        plan = F.plan_create()
        step = next(s for s in plan.steps
                    if s.kind == StepKind.POST_METHOD)
        self.assertEqual(step.method_name, 'set_tag')
        self.assertEqual(step.method_args, ('default-tag',))
        obj = plan.commit()
        self.assertEqual(obj.tag, 'default-tag')
        self.assertEqual(calls, ['default-tag'])

    def test_post_generation_method_call_override(self):
        class WithMethod(Model):
            def set_tag(self, tag=None):
                self.tag = tag

        class F(RecorderFactory):
            class Meta:
                model = WithMethod

            tag = factory.PostGenerationMethodCall('set_tag')

        obj = F.plan_create(tag='custom').commit()
        self.assertEqual(obj.tag, 'custom')

    def test_post_generation_hook_context(self):
        seen = []

        class F(PointedFactory):
            @factory.post_generation
            def note(self, create, value, **kwargs):
                seen.append((create, value, kwargs))

        plan = F.plan_create(note__x=1, note='v')
        hook = next(s for s in plan.steps if s.kind == StepKind.POST_HOOK)
        self.assertTrue(hook.hook_context.value_provided)
        self.assertEqual(hook.hook_context.value, 'v')
        self.assertEqual(hook.hook_context.extra, {'x': 1})
        plan.commit()
        self.assertEqual(seen, [(True, 'v', {'x': 1})])

    def test_custom_post_generation_declaration_is_replayed(self):
        from factory.declarations import PostGenerationDeclaration

        class CustomDecl(PostGenerationDeclaration):
            def call(self, instance, step, context):
                instance.custom = ('ran', context.value)
                return context.value

        class F(PointedFactory):
            thing = CustomDecl()

        obj = F.plan_create(thing='ok').commit()
        self.assertEqual(obj.custom, ('ran', 'ok'))


class EmptyRelationTests(unittest.TestCase):
    def setUp(self):
        creates.clear()
        PointedFactory.reset_sequence(0)

    def test_related_factory_with_provided_value_is_skipped(self):
        existing = Pointer(bar='existing')
        plan = PointedRelatedFactory.plan_create(pointer=existing)
        related = next(s for s in plan.steps
                       if s.declaration_name == 'pointer')
        self.assertEqual(related.kind, StepKind.RELATED)
        self.assertIsNone(related.related_child)
        self.assertEqual(related.status, StepStatus.SKIPPED)
        pointed = plan.commit()
        # No Pointer was generated; only the root was created.
        self.assertEqual([m for m, _ in creates], [Pointed])
        self.assertIs(plan.tree.post_children[0].result, existing)
        self.assertEqual(pointed.foo, 'foo0')

    def test_related_factory_with_none_is_empty(self):
        plan = PointedRelatedFactory.plan_create(pointer=None)
        related = next(s for s in plan.steps
                       if s.declaration_name == 'pointer')
        self.assertEqual(related.status, StepStatus.SKIPPED)
        plan.commit()
        self.assertEqual([m for m, _ in creates], [Pointed])

    def test_subfactory_overridden_with_none(self):
        plan = PointerFactory.plan_create(pointed=None)
        self.assertIsNone(plan.tree.kwargs['pointed'])
        ptr = plan.commit()
        self.assertIsNone(ptr.pointed)
        self.assertEqual([m for m, _ in creates], [Pointer])

    def test_related_factory_list_zero(self):
        class F(PointedFactory):
            pointers = factory.RelatedFactoryList(
                PointerFactory, factory_related_name='pointed', size=0)

        plan = F.plan_create()
        group = next(s for s in plan.steps
                     if s.kind == StepKind.RELATED_LIST)
        self.assertEqual(group.post_children, [])
        plan.commit()
        self.assertEqual([m for m, _ in creates], [Pointed])

    def test_related_factory_list_three(self):
        class F(PointedFactory):
            pointers = factory.RelatedFactoryList(
                PointerFactory, factory_related_name='pointed', size=3)

        plan = F.plan_create()
        group = next(s for s in plan.steps
                     if s.kind == StepKind.RELATED_LIST)
        self.assertEqual(len(group.post_children), 3)
        pointed = plan.commit()
        pointer_creates = [kw for m, kw in creates if m is Pointer]
        self.assertEqual(len(pointer_creates), 3)
        # Every related object receives the same parent placeholder,
        # substituted with the actual parent instance at execution.
        self.assertTrue(
            all(kw['pointed'] is pointed for kw in pointer_creates))

    def test_related_factory_list_callable_size(self):
        class F(PointedFactory):
            pointers = factory.RelatedFactoryList(
                PointerFactory, factory_related_name='pointed',
                size=lambda: 2)

        plan = F.plan_create()
        group = next(s for s in plan.steps
                     if s.kind == StepKind.RELATED_LIST)
        self.assertEqual(len(group.post_children), 2)

    def test_trait_off_skips_related_factory(self):
        class F(PointedFactory):
            class Params:
                with_pointer = factory.Trait(
                    pointer=factory.RelatedFactory(
                        PointerFactory, factory_related_name='pointed'))

        plan = F.plan_create(with_pointer=False)
        statuses = [(s.kind, s.status) for s in plan.steps
                    if s.declaration_name == 'pointer']
        self.assertTrue(any(k == StepKind.POST_MAYBE for k, _ in statuses))
        self.assertTrue(any(st == StepStatus.SKIPPED for _, st in statuses))
        plan.commit()
        self.assertEqual([m for m, _ in creates], [Pointed])

        # The second plan builds its own Pointed plus the related Pointer;
        # the two plans' rows accumulate independently.
        plan = F.plan_create(with_pointer=True)
        plan.commit()
        self.assertEqual([m for m, _ in creates],
                         [Pointed, Pointed, Pointer])
        # But within a single plan, the related Pointer reuses its parent
        # placeholder instead of spawning another Pointed subfactory.
        self.assertEqual(
            [m for m, _ in creates][1:], [Pointed, Pointer])


class PartialFailureTests(unittest.TestCase):
    def setUp(self):
        creates.clear()
        PointedFactory.reset_sequence(0)

    def test_failure_reports_completed_and_remaining(self):
        attempts = {'n': 0}

        class RelatedPointer(PointerFactory):
            @factory.post_generation
            def explode(self, create, value, **kwargs):
                attempts['n'] += 1
                if attempts['n'] == 1:
                    raise RuntimeError('boom in related')

        class F(PointedFactory):
            pointer = factory.RelatedFactory(
                RelatedPointer, factory_related_name='pointed')

        plan = F.plan_create()
        with self.assertRaises(PartialCommitError) as cm:
            plan.commit()

        error = cm.exception
        completed_models = [m for m, _ in creates]
        self.assertEqual(completed_models, [Pointed, Pointer])
        self.assertIn(plan.tree, error.completed)
        self.assertEqual(error.failed_step.kind, StepKind.POST_HOOK)
        self.assertGreater(len(error.remaining), 0)
        self.assertIsInstance(error.original, RuntimeError)
        self.assertIs(plan.failed_step, error.failed_step)

    def test_idempotent_retry_does_not_duplicate(self):
        attempts = {'n': 0}

        class RelatedPointer(PointerFactory):
            @factory.post_generation
            def explode(self, create, value, **kwargs):
                attempts['n'] += 1
                if attempts['n'] == 1:
                    raise RuntimeError('boom in related')

        class F(PointedFactory):
            pointer = factory.RelatedFactory(
                RelatedPointer, factory_related_name='pointed')

        plan = F.plan_create()
        with self.assertRaises(PartialCommitError):
            plan.commit()
        self.assertEqual([m for m, _ in creates], [Pointed, Pointer])
        pointed = plan.commit()
        self.assertEqual([m for m, _ in creates], [Pointed, Pointer])
        self.assertIs(pointed, plan.tree.result)
        self.assertTrue(plan.succeeded)
        self.assertIsNone(plan.failed_step)

    def test_failure_in_root_then_retry(self):
        attempts = {'n': 0}

        class FlakyPointed(PointedFactory):
            @classmethod
            def _create(cls, model_class, *args, **kwargs):
                attempts['n'] += 1
                if attempts['n'] == 1:
                    raise RuntimeError('cannot insert')
                return super()._create(model_class, *args, **kwargs)

        plan = FlakyPointed.plan_create()
        with self.assertRaises(PartialCommitError) as cm:
            plan.commit()
        self.assertEqual(cm.exception.completed, [])
        self.assertEqual(cm.exception.failed_step, plan.tree)
        self.assertEqual(creates, [])
        obj = plan.commit()
        self.assertEqual(obj.foo, 'foo0')
        self.assertEqual(len(creates), 1)

    def test_failure_after_root_skips_later_relations(self):
        class HookFactory(PointedFactory):
            @factory.post_generation
            def explode(self, create, value, **kwargs):
                raise RuntimeError('boom')

            late_pointer = factory.RelatedFactory(
                PointerFactory, factory_related_name='pointed')

        plan = HookFactory.plan_create()
        names = [p.declaration_name for p in plan.tree.post_children]
        self.assertLess(names.index('explode'), names.index('late_pointer'))
        with self.assertRaises(PartialCommitError):
            plan.commit()
        self.assertEqual([m for m, _ in creates], [Pointed])
        self.assertTrue(any(
            s.kind == StepKind.RELATED and s.status == StepStatus.PENDING
            for s in plan.steps))


# ---------------------------------------------------------------------------
# SQLAlchemy tests
# ---------------------------------------------------------------------------

if sqlalchemy is not None:
    from .alchemyapp import models as sa_models

    class SAParent(sa_models.Base):
        __tablename__ = 'plan_test_parent'
        id = Column(Integer, primary_key=True)
        name = Column(Unicode(20))
        children = relationship(
            'SAChild', back_populates='parent',
            foreign_keys='SAChild.parent_id')
        first_child_id = Column(
            Integer,
            ForeignKey('plan_test_child.id', use_alter=True),
            nullable=True)
        first_child = relationship(
            'SAChild', foreign_keys=[first_child_id])

    class SAChild(sa_models.Base):
        __tablename__ = 'plan_test_child'
        id = Column(Integer, primary_key=True)
        label = Column(Unicode(20))
        parent_id = Column(Integer, ForeignKey('plan_test_parent.id'))
        parent = relationship(
            'SAParent', back_populates='children',
            foreign_keys=[parent_id])

    class SATransactionTestCase(unittest.TestCase):
        def setUp(self):
            sa_models.Base.metadata.create_all(sa_models.engine)

        def tearDown(self):
            sa_models.session.remove()
            sa_models.Base.metadata.drop_all(sa_models.engine)

    class SAChildFactory(factory.alchemy.SQLAlchemyModelFactory):
        class Meta:
            model = SAChild
            sqlalchemy_session = sa_models.session
            sqlalchemy_session_persistence = 'commit'

        label = factory.Sequence(lambda n: 'c%d' % n)

    class SAParentFactory(factory.alchemy.SQLAlchemyModelFactory):
        class Meta:
            model = SAParent
            sqlalchemy_session = sa_models.session
            sqlalchemy_session_persistence = 'commit'

        name = 'parent'
        children = factory.RelatedFactoryList(
            SAChildFactory, factory_related_name='parent', size=2)

    class SQLAlchemyPlanTests(SATransactionTestCase):
        def test_preview_touches_no_session(self):
            plan = SAParentFactory.plan_create()
            self.assertEqual(list(sa_models.session.new), [])
            self.assertEqual(sa_models.session.query(SAParent).count(), 0)
            self.assertEqual(sa_models.session.query(SAChild).count(), 0)
            boundary_kinds = {b.kind for b in plan.transaction_boundaries}
            self.assertEqual(boundary_kinds, {'commit'})

        def test_commit_persists_related_factory_list(self):
            plan = SAParentFactory.plan_create()
            parent = plan.commit()
            self.assertEqual(sa_models.session.query(SAParent).count(), 1)
            self.assertEqual(sa_models.session.query(SAChild).count(), 2)
            self.assertEqual(sorted(c.label for c in parent.children),
                             ['c0', 'c1'])
            plan.commit()  # idempotent
            self.assertEqual(sa_models.session.query(SAChild).count(), 2)

        def test_managed_persistence_leaves_transaction_open(self):
            class ManagedFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAParent
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_session_persistence = None

                name = 'managed'

            plan = ManagedFactory.plan_create()
            self.assertEqual(
                [b.kind for b in plan.transaction_boundaries], ['managed'])
            plan.commit()
            self.assertIn(plan.instance, sa_models.session.new)
            sa_models.session.commit()
            self.assertEqual(sa_models.session.query(SAParent).count(), 1)
            sa_models.session.rollback()

        def test_flush_persistence_boundary(self):
            class FlushFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAParent
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_session_persistence = 'flush'

                name = 'flush'

            plan = FlushFactory.plan_create()
            self.assertEqual(
                [b.kind for b in plan.transaction_boundaries], ['flush'])
            plan.commit()
            self.assertEqual(sa_models.session.query(SAParent).count(), 1)
            sa_models.session.rollback()

        def test_partial_failure_then_retry_with_commit_persistence(self):
            state = {'n': 0}

            class FlakyFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAParent
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_session_persistence = 'commit'

                name = 'flaky'

                @factory.post_generation
                def explode(self, create, value, **kwargs):
                    state['n'] += 1
                    if state['n'] == 1:
                        raise RuntimeError('boom')

            state['n'] = 0
            plan = FlakyFactory.plan_create()
            with self.assertRaises(PartialCommitError) as cm:
                plan.commit()
            self.assertEqual(cm.exception.failed_step.kind,
                             StepKind.POST_HOOK)
            self.assertEqual(sa_models.session.query(SAParent).count(), 1)
            plan.commit()
            self.assertEqual(sa_models.session.query(SAParent).count(), 1)

        def test_flush_failure_session_reuse_and_retry(self):
            state = {'n': 0}

            class FlakyFlushFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAParent
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_session_persistence = 'flush'

                name = 'flaky-flush'

                @factory.post_generation
                def explode(self, create, value, **kwargs):
                    state['n'] += 1
                    if state['n'] == 1:
                        # A non-database error does not invalidate the session.
                        raise ValueError('not a DB error')

            state['n'] = 0
            plan = FlakyFlushFactory.plan_create()
            with self.assertRaises(PartialCommitError):
                plan.commit()
            plan.commit()  # root node is done, only the hook reruns
            sa_models.session.commit()
            self.assertEqual(
                sa_models.session.query(SAParent)
                .filter_by(name='flaky-flush').count(),
                1)

        def test_get_or_create_retry_is_idempotent(self):
            state = {'n': 0}

            class GOCFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = sa_models.MultiFieldModel
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_get_or_create = ('slug',)
                    sqlalchemy_session_persistence = 'commit'

                id = factory.Sequence(lambda n: n + 1)
                foo = 'f'
                slug = 'the-slug'

                @factory.post_generation
                def explode(self, create, value, **kwargs):
                    state['n'] += 1
                    if state['n'] == 1:
                        raise RuntimeError('boom')

            state['n'] = 0
            plan = GOCFactory.plan_create()
            with self.assertRaises(PartialCommitError):
                plan.commit()
            self.assertEqual(
                sa_models.session.query(sa_models.MultiFieldModel)
                .filter_by(slug='the-slug').count(),
                1)
            plan.commit()
            self.assertEqual(
                sa_models.session.query(sa_models.MultiFieldModel)
                .filter_by(slug='the-slug').count(),
                1)

        def test_session_factory_not_invoked_during_planning(self):
            created_sessions = []

            def session_factory():
                new_session = sa_models.session()
                created_sessions.append(new_session)
                return new_session

            class SessionFactoryFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAParent
                    sqlalchemy_session_factory = staticmethod(session_factory)
                    sqlalchemy_session_persistence = 'commit'

                name = 'sf'

            try:
                plan = SessionFactoryFactory.plan_create()
                self.assertEqual(created_sessions, [])
                plan.commit()
                self.assertEqual(len(created_sessions), 1)
            finally:
                SessionFactoryFactory._meta.sqlalchemy_session_factory = None
                SessionFactoryFactory._meta.sqlalchemy_session = sa_models.session

        def test_empty_relation_none(self):
            class NullableChildFactory(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAChild
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_session_persistence = 'commit'

                label = 'solo'

            class ParentWithOptional(factory.alchemy.SQLAlchemyModelFactory):
                class Meta:
                    model = SAParent
                    sqlalchemy_session = sa_models.session
                    sqlalchemy_session_persistence = 'commit'

                name = 'p'
                first_child = factory.SubFactory(NullableChildFactory)

            plan = ParentWithOptional.plan_create(first_child=None)
            self.assertIsNone(plan.tree.kwargs['first_child'])
            plan.commit()
            self.assertEqual(sa_models.session.query(SAChild).count(), 0)
            self.assertEqual(sa_models.session.query(SAParent).count(), 1)


# ---------------------------------------------------------------------------
# Django tests
# ---------------------------------------------------------------------------

if django is not None:
    from django.db.models import signals as django_signals
    from django.test import utils as django_test_utils

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.djapp.settings')
    django.setup()

    from .djapp import models as dj_models  # noqa: E402

    _django_state = {}

    def setUpModule():
        django_test_utils.setup_test_environment()
        _django_state['runner'] = django_test_utils.setup_databases(
            verbosity=0, interactive=False)

    def tearDownModule():
        django_test_utils.teardown_databases(
            _django_state['runner'], verbosity=0)
        django_test_utils.teardown_test_environment()

    class DjPointedFactory(factory.django.DjangoModelFactory):
        class Meta:
            model = dj_models.PointedModel
            skip_postgeneration_save = True

        foo = 'foo'

    class DjPointerFactory(factory.django.DjangoModelFactory):
        class Meta:
            model = dj_models.PointerModel

        bar = 'bar'
        pointed = factory.SubFactory(DjPointedFactory)

    class DjPointedRelatedFactory(DjPointedFactory):
        pointer = factory.RelatedFactory(
            DjPointerFactory, factory_related_name='pointed')

    class DjangoPlanTests(django.test.TransactionTestCase):
        def test_preview_does_not_touch_database(self):
            plan = DjPointedRelatedFactory.plan_create()
            self.assertEqual(dj_models.PointedModel.objects.count(), 0)
            self.assertEqual(dj_models.PointerModel.objects.count(), 0)
            kinds = [s.kind for s in plan.steps]
            self.assertIn(StepKind.INSTANTIATE, kinds)
            self.assertIn(StepKind.RELATED, kinds)
            for boundary in plan.transaction_boundaries:
                self.assertEqual(boundary.kind, 'autocommit')

        def test_commit_matches_normal_create(self):
            plan = DjPointedRelatedFactory.plan_create(pointer__bar='deep')
            pointed = plan.commit()
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)
            pointer = dj_models.PointerModel.objects.get()
            self.assertEqual(pointer.bar, 'deep')
            self.assertEqual(pointer.pointed_id, pointed.pk)
            plan.commit()  # idempotent
            self.assertEqual(dj_models.PointerModel.objects.count(), 1)

        def test_boundaries_describe_get_or_create_and_resave(self):
            class GOCFactory(factory.django.DjangoModelFactory):
                class Meta:
                    model = dj_models.MultifieldModel
                    django_get_or_create = ('slug',)

                slug = 'sluggy'
                text = 'hi'

                @factory.post_generation
                def touched(self, create, value, **kwargs):
                    pass

            plan = GOCFactory.plan_create()
            labels = [b.label for b in plan.transaction_boundaries]
            self.assertTrue(any('get_or_create' in label for label in labels))
            self.assertTrue(any('save() UPDATE' in label for label in labels))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                obj = plan.commit()
            self.assertEqual(obj.slug, 'sluggy')

        def test_skip_postgeneration_save_has_no_resave_boundary(self):
            plan = DjPointedRelatedFactory.plan_create()
            labels = [b.label for b in plan.transaction_boundaries]
            self.assertFalse(
                any('save() UPDATE' in label for label in labels))

        def test_empty_relations(self):
            plan = DjPointedRelatedFactory.plan_create(pointer=None)
            plan.commit()
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)
            self.assertEqual(dj_models.PointerModel.objects.count(), 0)

            plan = DjPointerFactory.plan_create(pointed=None)
            ptr = plan.commit()
            self.assertIsNone(ptr.pointed_id)
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)

        def test_related_factory_with_provided_value(self):
            unsaved = DjPointerFactory.build(bar='unsaved')
            plan = DjPointedRelatedFactory.plan_create(pointer=unsaved)
            related = next(s for s in plan.steps
                           if s.declaration_name == 'pointer')
            self.assertIsNone(related.related_child)
            self.assertEqual(dj_models.PointerModel.objects.count(), 0)
            plan.commit()
            # A built (unsaved) value is used as-is: no pointer persisted.
            self.assertEqual(dj_models.PointerModel.objects.count(), 0)
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)

        def test_partial_failure_report_and_retry(self):
            state = {'n': 0}

            class FlakyRelated(DjPointerFactory):
                @factory.post_generation
                def explode(self, create, value, **kwargs):
                    state['n'] += 1
                    if state['n'] == 1:
                        raise RuntimeError('boom')

            class FlakyPointed(DjPointedFactory):
                pointer = factory.RelatedFactory(
                    FlakyRelated, factory_related_name='pointed')

            state['n'] = 0
            plan = FlakyPointed.plan_create()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                with self.assertRaises(PartialCommitError) as cm:
                    plan.commit()
            err = cm.exception
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)
            self.assertEqual(dj_models.PointerModel.objects.count(), 1)
            self.assertEqual(err.failed_step.declaration_name, 'explode')
            completed_ids = [s.id for s in err.completed]
            self.assertIn(plan.tree.id, completed_ids)
            remaining_names = [
                s.declaration_name for s in err.remaining
                if s.declaration_name is not None]
            self.assertIn('pointer', remaining_names)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                pointed = plan.commit()
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)
            self.assertEqual(dj_models.PointerModel.objects.count(), 1)
            self.assertEqual(pointed.pointer.bar, 'bar')

        def test_failed_related_factory_can_be_retried(self):
            state = {'n': 0}

            class FlakyPointer(DjPointerFactory):
                @classmethod
                def _create(cls, model_class, *args, **kwargs):
                    state['n'] += 1
                    if state['n'] == 1:
                        raise RuntimeError('insert failed')
                    return super()._create(model_class, *args, **kwargs)

            class FlakyPointed(DjPointedFactory):
                pointer = factory.RelatedFactory(
                    FlakyPointer, factory_related_name='pointed')

            plan = FlakyPointed.plan_create()
            with self.assertRaises(PartialCommitError):
                plan.commit()
            self.assertEqual(dj_models.PointedModel.objects.count(), 1)
            self.assertEqual(dj_models.PointerModel.objects.count(), 0)
            plan.commit()
            self.assertEqual(dj_models.PointerModel.objects.count(), 1)
            self.assertEqual(state['n'], 2)

        def test_get_or_create_idempotent_across_failure(self):
            state = {'n': 0}

            class GOCFactory(factory.django.DjangoModelFactory):
                class Meta:
                    model = dj_models.MultifieldModel
                    django_get_or_create = ('slug',)
                    skip_postgeneration_save = True

                slug = 'goc-slug'
                text = 'x'

                @factory.post_generation
                def explode(self, create, value, **kwargs):
                    state['n'] += 1
                    if state['n'] == 1:
                        raise RuntimeError('boom')

            state['n'] = 0
            plan = GOCFactory.plan_create()
            with self.assertRaises(PartialCommitError):
                plan.commit()
            plan.commit()
            self.assertEqual(
                dj_models.MultifieldModel.objects.filter(
                    slug='goc-slug').count(),
                1)

        def test_sequence_and_iterator_unchanged_by_preview(self):
            class SeqFactory(factory.django.DjangoModelFactory):
                class Meta:
                    model = dj_models.StandardModel
                    skip_postgeneration_save = True

                foo = factory.Sequence(lambda n: 's%d' % n)

            SeqFactory.reset_sequence(10)
            plan = SeqFactory.plan_create()
            self.assertEqual(plan.tree.kwargs['foo'], 's10')
            obj = SeqFactory.create()
            self.assertEqual(obj.foo, 's10')

            class IterFactory(factory.django.DjangoModelFactory):
                class Meta:
                    model = dj_models.StandardModel
                    skip_postgeneration_save = True

                foo = factory.Iterator(['a', 'b'])

            IterFactory.plan_create()
            self.assertEqual(IterFactory.create().foo, 'a')

    class MuteSignalsPlanTests(django.test.TransactionTestCase):
        def setUp(self):
            self.received = []

            def receiver(sender, instance, **kwargs):
                self.received.append(instance)

            self.receiver = receiver
            django_signals.post_save.connect(
                receiver, sender=dj_models.PointedModel)

        def tearDown(self):
            django_signals.post_save.disconnect(
                self.receiver, sender=dj_models.PointedModel)

        def test_plan_describes_muted_signals(self):
            @factory.django.mute_signals(django_signals.post_save)
            class MutedFactory(DjPointedFactory):
                pass

            plan = MutedFactory.plan_create()
            self.assertIn(django_signals.post_save, plan.muted_signals)
            # Planning does not fire signals.
            self.assertEqual(self.received, [])
            DjPointedFactory.create()
            self.assertEqual(len(self.received), 1)

        def test_commit_mutes_signals(self):
            @factory.django.mute_signals(django_signals.post_save)
            class MutedFactory(DjPointedFactory):
                pass

            plan = MutedFactory.plan_create()
            plan.commit()
            self.assertEqual(self.received, [])
            # Signals restored after commit.
            DjPointedFactory.create()
            self.assertEqual(len(self.received), 1)

        def test_muting_restored_after_failure(self):
            @factory.django.mute_signals(django_signals.post_save)
            class MutedFactory(DjPointedFactory):
                @factory.post_generation
                def explode(self, create, value, **kwargs):
                    raise RuntimeError('boom')

            plan = MutedFactory.plan_create()
            with self.assertRaises(PartialCommitError):
                plan.commit()
            self.assertEqual(self.received, [])
            # Signal receivers must be restored despite the failure.
            DjPointedFactory.create()
            self.assertEqual(len(self.received), 1)

        def test_muted_related_factory_step_is_marked(self):
            pointer_received = []

            def pointer_receiver(sender, instance, **kwargs):
                pointer_received.append(instance)

            @factory.django.mute_signals(django_signals.post_save)
            class MutedPointer(DjPointerFactory):
                pass

            class F(DjPointedFactory):
                pointer = factory.RelatedFactory(
                    MutedPointer, factory_related_name='pointed')

            django_signals.post_save.connect(
                pointer_receiver, sender=dj_models.PointerModel)
            try:
                plan = F.plan_create()
                related_child = next(
                    s for s in plan.steps
                    if s.kind == StepKind.INSTANTIATE
                    and s.model is dj_models.PointerModel)
                self.assertTrue(related_child.muted_signals)
                plan.commit()
                # The root (unmuted) fires post_save for PointedModel...
                self.assertEqual(len(self.received), 1)
                # ...but the muted related Pointer does not.
                self.assertEqual(pointer_received, [])
            finally:
                django_signals.post_save.disconnect(
                    pointer_receiver, sender=dj_models.PointerModel)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

# Copyright: See the LICENSE file.

"""Tests for the post-generation result aggregator and reports."""


import dataclasses
import unittest

import factory
from factory import builder, declarations, errors, postgeneration


class Obj:
    def __init__(self):
        self.results = None


class Related:
    def __init__(self, parent=None):
        self.parent = parent


class EmptyReportTestCase(unittest.TestCase):
    def test_no_post_declarations(self):
        class EmptyFactory(factory.Factory):
            class Meta:
                model = Obj

        with factory.collect_postgeneration_reports() as collected:
            obj = EmptyFactory.build()

        report = collected.report
        self.assertIsInstance(obj, Obj)
        self.assertEqual(0, len(report))
        self.assertEqual(report, factory.PostGenerationReport.EMPTY)
        self.assertTrue(report.ok)
        self.assertFalse(report.interrupted)
        self.assertEqual((), report.entries)
        self.assertEqual((), report.names)
        self.assertEqual((), report.executed)
        self.assertEqual((), report.skipped)
        self.assertEqual((), report.failed)
        self.assertEqual({}, report.return_values)
        report.raise_if_failed()  # no-op

    def test_collector_starts_empty_and_resets(self):
        with factory.collect_postgeneration_reports() as collected:
            self.assertEqual(0, len(collected.reports))
            self.assertEqual(factory.PostGenerationReport.EMPTY, collected.report)

        self.assertIsNone(postgeneration.get_current_collector())


class MultiHookReportTestCase(unittest.TestCase):
    def test_multiple_hooks_order_statuses_and_values(self):
        calls = []

        class MultiHookFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def first(self, _create, _extracted):
                calls.append('first')
                return 1

            @factory.post_generation
            def second(self, _create, _extracted):
                calls.append('second')
                return None

            @factory.post_generation
            def third(self, _create, _extracted):
                calls.append('third')
                return 'three'

            @classmethod
            def _after_postgeneration(cls, obj, create, results):
                obj.results = results

        with factory.collect_postgeneration_reports() as collected:
            obj = MultiHookFactory.build()

        self.assertIsInstance(obj, Obj)
        self.assertEqual(['first', 'second', 'third'], calls)

        report = collected.report
        self.assertEqual(('first', 'second', 'third'), report.names)
        self.assertEqual(
            [factory.PostGenerationStatus.EXECUTED] * 3,
            [entry.status for entry in report],
        )
        self.assertEqual((0, 1, 2), tuple(entry.index for entry in report))
        self.assertTrue(report.ok)
        self.assertEqual({'first': 1, 'second': None, 'third': 'three'},
                         report.return_values)
        self.assertEqual(1, report['first'].value)
        self.assertIsNone(report['second'].value)
        self.assertEqual('three', report[2].value)
        self.assertIn('second', report)
        self.assertNotIn('missing', report)
        self.assertTrue(all(entry.was_executed for entry in report))
        self.assertFalse(any(entry.failed for entry in report))

        # The legacy results mapping passed to _after_postgeneration is
        # unchanged and still maps every declaration to its raw return value.
        self.assertEqual({'first': 1, 'second': None, 'third': 'three'},
                         obj.results)

        with self.assertRaises(KeyError):
            report['missing']
        with self.assertRaises(IndexError):
            report[42]

    def test_create_strategy_records_create_flag(self):
        class CreateFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def hook(self, create, _extracted):
                return create

        with factory.collect_postgeneration_reports() as collected:
            CreateFactory.create()
            CreateFactory.build()

        records = collected.records
        self.assertTrue(records[0].create)
        self.assertFalse(records[1].create)
        self.assertTrue(records[0].report['hook'].value)
        self.assertFalse(records[1].report['hook'].value)


class SkippedDeclarationTestCase(unittest.TestCase):
    def test_maybe_skips_post_generation(self):
        calls = []

        def hook(obj, create, extracted):
            calls.append('hook')
            return 'hooked'

        class MaybeFactory(factory.Factory):
            class Meta:
                model = Obj

            class Params:
                enable_hook = False

            optional = factory.Maybe(
                'enable_hook',
                factory.PostGeneration(hook),
            )

            @classmethod
            def _after_postgeneration(cls, obj, create, results):
                obj.results = results

        with factory.collect_postgeneration_reports() as collected:
            obj_disabled = MaybeFactory.build()

        report = collected.report
        self.assertEqual(('optional',), report.names)
        self.assertEqual(factory.PostGenerationStatus.SKIPPED,
                         report['optional'].status)
        self.assertTrue(report['optional'].was_skipped)
        self.assertIsNone(report['optional'].value)
        self.assertEqual([], calls)
        self.assertEqual({}, report.return_values)
        # Legacy results still carry the SKIP marker, as before.
        self.assertIs(declarations.SKIP, obj_disabled.results['optional'])

        with factory.collect_postgeneration_reports() as collected:
            MaybeFactory.build(enable_hook=True)

        report = collected.report
        self.assertEqual(factory.PostGenerationStatus.EXECUTED,
                         report['optional'].status)
        self.assertEqual('hooked', report['optional'].value)
        self.assertEqual(['hook'], calls)


class RelatedFactoryReportTestCase(unittest.TestCase):
    def test_related_factory_is_executed_once_and_nested_report_collected(self):
        related_objects = []

        class TrackedRelated:
            def __init__(self, parent=None):
                self.parent = parent
                related_objects.append(self)

        class TrackedRelatedFactory(factory.Factory):
            class Meta:
                model = TrackedRelated

            @factory.post_generation
            def tag(self, _create, _extracted):
                return 'tagged'

        class OwningFactory(factory.Factory):
            class Meta:
                model = Obj

            child = factory.RelatedFactory(
                TrackedRelatedFactory,
                factory_related_name='parent',
            )

        with factory.collect_postgeneration_reports() as collected:
            obj = OwningFactory.build()

        self.assertEqual(1, len(related_objects))
        self.assertIs(obj, related_objects[0].parent)

        # Nested (RelatedFactory) report completes first, root report last.
        factories = [record.factory for record in collected.records]
        self.assertEqual(TrackedRelatedFactory, factories[0])
        self.assertEqual(OwningFactory, factories[-1])

        root_report = collected.report
        self.assertEqual(('child',), root_report.names)
        child_result = root_report['child']
        self.assertEqual(factory.PostGenerationStatus.EXECUTED,
                         child_result.status)
        self.assertIs(related_objects[0], child_result.value)

        nested = collected.for_factory(TrackedRelatedFactory)
        self.assertEqual(1, len(nested))
        self.assertEqual('tagged', nested[0]['tag'].value)
        self.assertEqual((root_report,), collected.for_factory(OwningFactory))

    def test_continue_policy_keeps_related_factory_value_when_nested_hook_fails(self):
        class FailingRelatedFactory(factory.Factory):
            class Meta:
                model = Related

            @factory.post_generation
            def boom(self, _create, _extracted):
                raise RuntimeError('nested failure')

        class OwningFactory(factory.Factory):
            class Meta:
                model = Obj

            child = factory.RelatedFactory(
                FailingRelatedFactory,
                factory_related_name='parent',
            )

        with factory.collect_postgeneration_reports(
                on_error=factory.PostGenerationAggregator.CONTINUE) as collected:
            obj = OwningFactory.build()

        nested_report = collected.for_factory(FailingRelatedFactory)[0]
        self.assertEqual(factory.PostGenerationStatus.FAILED,
                         nested_report['boom'].status)

        # The RelatedFactory still returned its generated instance.
        self.assertIsInstance(obj, Obj)
        self.assertIsInstance(
            collected.report['child'].value, Related,
        )


class FailurePolicyTestCase(unittest.TestCase):
    def test_default_behaviour_propagates_original_exception_without_wrapper(self):
        calls = []

        class FailingFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def first(self, _create, _extracted):
                calls.append('first')
                return 1

            @factory.post_generation
            def second(self, _create, _extracted):
                calls.append('second')
                raise ValueError('boom')

            @factory.post_generation
            def third(self, _create, _extracted):
                calls.append('third')

        with self.assertRaises(ValueError) as cm:
            FailingFactory.build()

        self.assertEqual('boom', str(cm.exception))
        self.assertEqual(['first', 'second'], calls)

    def test_stop_policy_preserves_executed_boundary(self):
        calls = []

        class FailingFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def first(self, _create, _extracted):
                calls.append('first')
                return 1

            @factory.post_generation
            def second(self, _create, _extracted):
                calls.append('second')
                raise ValueError('boom')

            @factory.post_generation
            def third(self, _create, _extracted):
                calls.append('third')
                return 3

            @classmethod
            def _after_postgeneration(cls, obj, create, results):
                obj.results = results

        with factory.collect_postgeneration_reports() as collected:
            with self.assertRaises(ValueError) as cm:
                FailingFactory.build()

        report = collected.report
        self.assertTrue(report.interrupted)
        self.assertEqual(('first', 'second'), report.names)
        self.assertNotIn('third', report)
        self.assertEqual(factory.PostGenerationStatus.EXECUTED,
                         report['first'].status)
        self.assertEqual(1, report['first'].value)
        failed = report['second']
        self.assertEqual(factory.PostGenerationStatus.FAILED, failed.status)
        self.assertIs(cm.exception, failed.exception)
        self.assertIs(cm.exception, report.failed[0].exception)
        self.assertFalse(report.ok)
        self.assertEqual(['first', 'second'], calls)

    def test_continue_policy_records_failure_and_runs_remaining_hooks(self):
        calls = []

        class FailingFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def first(self, _create, _extracted):
                calls.append('first')
                return 1

            @factory.post_generation
            def second(self, _create, _extracted):
                calls.append('second')
                raise ValueError('boom')

            @factory.post_generation
            def third(self, _create, _extracted):
                calls.append('third')
                return 3

            @classmethod
            def _after_postgeneration(cls, obj, create, results):
                obj.results = results

        with factory.collect_postgeneration_reports(
                on_error=factory.PostGenerationAggregator.CONTINUE) as collected:
            obj = FailingFactory.create()

        self.assertIsInstance(obj, Obj)
        self.assertEqual(['first', 'second', 'third'], calls)

        report = collected.report
        self.assertFalse(report.interrupted)
        self.assertEqual(
            ('executed', 'failed', 'executed'),
            tuple(entry.status for entry in report),
        )
        self.assertIsInstance(report.failed[0].exception, ValueError)
        self.assertEqual('boom', report.failed[0].exception.args[0])
        self.assertEqual({'first': 1, 'third': 3}, report.return_values)
        # Failed declaration is absent from the legacy results mapping.
        self.assertEqual({'first': 1, 'third': 3}, obj.results)

        with self.assertRaises(errors.PostGenerationError) as cm:
            report.raise_if_failed()
        self.assertIs(report, cm.exception.report)
        self.assertEqual('second', cm.exception.declaration)
        self.assertIs(report.failed[0].exception, cm.exception.__cause__)

    def test_hooks_never_run_twice(self):
        calls = []

        class FailOnceFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def first(self, _create, _extracted):
                calls.append('first')
                return 1

            @factory.post_generation
            def second(self, _create, _extracted):
                calls.append('second')
                raise ValueError('boom')

            @factory.post_generation
            def third(self, _create, _extracted):
                calls.append('third')

        with factory.collect_postgeneration_reports(
                on_error=factory.PostGenerationAggregator.CONTINUE):
            FailOnceFactory.build()
            FailOnceFactory.build()

        # Two builds: each hook ran once per build, never re-executed to
        # recover from the failure.
        self.assertEqual(
            ['first', 'second', 'third', 'first', 'second', 'third'],
            calls,
        )


class ImmutabilityTestCase(unittest.TestCase):
    def test_report_and_results_are_immutable(self):
        class ImmutableFactory(factory.Factory):
            class Meta:
                model = Obj

            @factory.post_generation
            def hook(self, _create, _extracted):
                return 1

        with factory.collect_postgeneration_reports() as collected:
            ImmutableFactory.build()
        report = collected.report

        with self.assertRaises(dataclasses.FrozenInstanceError):
            report.interrupted = True
        with self.assertRaises(dataclasses.FrozenInstanceError):
            report.entries = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            report['hook'].value = 2
        with self.assertRaises(TypeError):
            report.return_values['other'] = 2
        with self.assertRaises(AttributeError):
            report.entries.append(None)


class AggregatorUnitTestCase(unittest.TestCase):
    class FakeDeclaration:
        def __init__(self, outcome):
            self.outcome = outcome
            self.calls = 0

        def evaluate_post(self, instance, step, overrides):
            self.calls += 1
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    def test_empty_declaration_set(self):
        aggregator = postgeneration.PostGenerationAggregator()
        outcome = aggregator.aggregate(builder.DeclarationSet(), None, None)
        self.assertEqual({}, outcome.results)
        self.assertEqual(factory.PostGenerationReport.EMPTY, outcome.report)

    def test_executed_skipped_failed_and_order(self):
        executed = self.FakeDeclaration('value')
        skipped = self.FakeDeclaration(declarations.SKIP)
        failed = self.FakeDeclaration(ValueError('nope'))
        later = self.FakeDeclaration('later')

        posts = builder.DeclarationSet({
            'executed': executed,
            'skipped': skipped,
            'failed': failed,
            'later': later,
        })

        aggregator = postgeneration.PostGenerationAggregator(
            on_error=postgeneration.PostGenerationAggregator.CONTINUE,
        )
        outcome = aggregator.aggregate(posts, instance=None, step=None)

        self.assertEqual(('executed', 'skipped', 'failed', 'later'),
                         outcome.report.names)
        self.assertEqual('value', outcome.report['executed'].value)
        self.assertTrue(outcome.report['skipped'].was_skipped)
        self.assertIsInstance(outcome.report['failed'].exception, ValueError)
        self.assertEqual('later', outcome.report['later'].value)
        self.assertEqual({'executed': 'value', 'skipped': declarations.SKIP,
                          'later': 'later'}, outcome.results)
        self.assertEqual(1, executed.calls)
        self.assertEqual(1, skipped.calls)
        self.assertEqual(1, failed.calls)
        self.assertEqual(1, later.calls)

    def test_stop_boundary_at_unit_level(self):
        before = self.FakeDeclaration(1)
        failing = self.FakeDeclaration(KeyError('x'))
        after = self.FakeDeclaration(2)

        posts = builder.DeclarationSet({
            'before': before,
            'failing': failing,
            'after': after,
        })

        outcome = postgeneration.PostGenerationAggregator().aggregate(
            posts, None, None,
        )
        self.assertTrue(outcome.report.interrupted)
        self.assertEqual(('before', 'failing'), outcome.report.names)
        self.assertEqual(1, before.calls)
        self.assertEqual(1, failing.calls)
        self.assertEqual(0, after.calls)

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            postgeneration.PostGenerationAggregator(on_error='restart')

    def test_status_values(self):
        self.assertEqual('skipped', factory.PostGenerationStatus.SKIPPED)
        self.assertEqual('executed', factory.PostGenerationStatus.EXECUTED)
        self.assertEqual('failed', factory.PostGenerationStatus.FAILED)


if __name__ == '__main__':
    unittest.main()

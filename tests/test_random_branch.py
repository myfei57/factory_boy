# Copyright: See the LICENSE file.

"""Tests for the named random-state branches (factory.random.random_branch)."""

import random
import threading
import unittest

import faker.generator
import faker.providers

import factory
from factory import fuzzy, random as factory_random
from factory.errors import RandomBranchError


SEED = 20240517


def _global_states():
    return (
        factory_random.randgen.getstate(),
        faker.generator.random.getstate(),
    )


def _fuzzy_value():
    return fuzzy.FuzzyInteger(0, 10 ** 9).fuzz()


def _faker_name():
    return factory.Faker('name').evaluate(None, None, {'locale': None})


def _draw_both():
    return _fuzzy_value(), _faker_name()


class RandomBranchTests(unittest.TestCase):
    def setUp(self):
        factory_random.reseed_random(SEED)
        self._initial_states = _global_states()
        # Tests may register custom faker providers; isolate that registry.
        self._real_fakers = factory.Faker._FAKER_REGISTRY.copy()
        self._real_providers = factory.Faker._CUSTOM_PROVIDERS.copy()

    def tearDown(self):
        factory.Faker._FAKER_REGISTRY = self._real_fakers
        factory.Faker._CUSTOM_PROVIDERS = self._real_providers
        # No test should leave a branch active in the main thread.
        self.assertIsNone(factory_random.get_active_branch())
        factory_random.reseed_random(SEED)

    # -- Basic isolation / repeatability -----------------------------------

    def test_branch_draws_do_not_touch_globals_while_active(self):
        g0, f0 = _global_states()
        with factory_random.random_branch('no-touch'):
            _draw_both()
            _fuzzy_value()
            _faker_name()
            # Mid-block, the global generators are exactly as on entry.
            self.assertEqual(g0, factory_random.randgen.getstate())
            self.assertEqual(f0, faker.generator.random.getstate())
        # After discard (the default), still unchanged.
        self.assertEqual((g0, f0), _global_states())

    def test_branch_results_are_repeatable(self):
        def run():
            values = []
            with factory_random.random_branch('replay'):
                values.extend(_draw_both())
                values.append(fuzzy.FuzzyText(length=8).fuzz())
                values.append(
                    fuzzy.FuzzyChoice(['a', 'b', 'c', 'd']).fuzz())
            return values

        first = run()
        factory_random.reseed_random(SEED)
        second = run()
        self.assertEqual(first, second)

    def test_branch_fuzzy_matches_replay_from_entry_state(self):
        with factory_random.random_branch('replay-fuzzy') as branch:
            entry_state = branch.randgen.getstate()
            values = [_fuzzy_value() for _ in range(10)]

        replay = random.Random()
        replay.setstate(entry_state)
        expected = [replay.randrange(0, 10 ** 9 + 1) for _ in range(10)]
        self.assertEqual(expected, values)

    def test_branch_faker_matches_replay_from_entry_state(self):
        with factory_random.random_branch('replay-faker') as branch:
            entry_state = branch.faker_random.getstate()
            values = [_faker_name() for _ in range(5)]

        replay = faker.Faker()
        replay.random = random.Random()
        replay.random.setstate(entry_state)
        self.assertEqual([replay.name() for _ in range(5)], values)

    def test_parent_stream_resumes_undisturbed_after_discard(self):
        factory_random.reseed_random(SEED)
        parent_before = [_draw_both() for _ in range(3)]

        factory_random.reseed_random(SEED)
        with factory_random.random_branch('detour'):
            [_draw_both() for _ in range(20)]
        parent_after = [_draw_both() for _ in range(3)]

        self.assertEqual(parent_before, parent_after)

    # -- Commit / discard ---------------------------------------------------

    def test_commit_advances_parent_as_if_draws_happened_there(self):
        with factory_random.random_branch('committed') as branch:
            branch_values = [_draw_both() for _ in range(6)]
            branch.commit()
        committed_states = _global_states()

        factory_random.reseed_random(SEED)
        direct_values = [_draw_both() for _ in range(6)]
        direct_states = _global_states()

        self.assertEqual(branch_values, direct_values)
        self.assertEqual(direct_states, committed_states)

    def test_discard_leaves_parent_state_untouched(self):
        with factory_random.random_branch('discarded') as branch:
            [_draw_both() for _ in range(6)]
            branch.discard()
        self.assertEqual(self._initial_states, _global_states())

    def test_default_exit_without_decision_discards(self):
        with factory_random.random_branch('default'):
            [_draw_both() for _ in range(6)]
        self.assertEqual(self._initial_states, _global_states())

    def test_empty_branch_commit_and_discard_are_noops(self):
        with factory_random.random_branch('empty-commit') as branch:
            branch.commit()
        self.assertEqual(self._initial_states, _global_states())

        with factory_random.random_branch('empty-discard') as branch:
            branch.discard()
        self.assertEqual(self._initial_states, _global_states())

        with factory_random.random_branch('empty-default'):
            pass
        self.assertEqual(self._initial_states, _global_states())

    # -- Nesting ------------------------------------------------------------

    def test_nested_branch_forks_from_enclosing_branch(self):
        with factory_random.random_branch('outer') as outer:
            outer_start = outer.randgen.getstate()
            _fuzzy_value()
            with factory_random.random_branch('inner') as inner:
                # The inner branch forks the outer's *current* state.
                self.assertIs(inner.parent, outer)
                self.assertEqual(
                    outer.randgen.getstate(), inner.randgen.getstate())
                inner_value = _fuzzy_value()
                inner.discard()
            # Discarded inner branch: the outer stream simply continues.
            replay = random.Random()
            replay.setstate(outer_start)
            replay.randrange(0, 10 ** 9 + 1)  # the outer draw
            expected_inner = replay.randrange(0, 10 ** 9 + 1)
            next_outer = _fuzzy_value()
            self.assertEqual(expected_inner, inner_value)
            self.assertEqual(expected_inner, next_outer)
            outer.commit()

    def test_nested_commit_propagates_to_enclosing_branch(self):
        with factory_random.random_branch('outer') as outer:
            with factory_random.random_branch('inner') as inner:
                inner_values = [_draw_both() for _ in range(4)]
                inner.commit()
            # Inner commits into the outer branch only: globals are still
            # untouched.
            self.assertEqual(self._initial_states, _global_states())
            outer_values_after = [_draw_both() for _ in range(4)]
            outer.discard()

        # Replay: forking a fresh branch at the same point and drawing 8
        # pairs must yield the inner values followed by the outer values.
        with factory_random.random_branch('replay'):
            replay_values = [_draw_both() for _ in range(8)]
        self.assertEqual(inner_values + outer_values_after, replay_values)

    def test_inner_exception_discards_inner_but_keeps_outer_alive(self):
        with factory_random.random_branch('outer') as outer:
            outer_state = outer.randgen.getstate()
            with self.assertRaises(RuntimeError):
                with factory_random.random_branch('inner'):
                    [_draw_both() for _ in range(5)]
                    raise RuntimeError('boom')
            # Inner is gone, outer is still the active branch and untouched.
            self.assertIs(factory_random.get_active_branch(), outer)
            self.assertEqual(outer_state, outer.randgen.getstate())
            outer.discard()
        self.assertEqual(self._initial_states, _global_states())

    # -- Named branches -----------------------------------------------------

    def test_duplicate_active_name_is_rejected(self):
        with factory_random.random_branch('dup'):
            with self.assertRaises(RandomBranchError):
                with factory_random.random_branch('dup'):
                    self.fail('duplicate branch should not be entered')
            # The rejected branch is not on the stack; parent still active.
            self.assertEqual('dup', factory_random.get_active_branch().name)

        # The name can be reused again once the branch is closed.
        with factory_random.random_branch('dup') as branch:
            self.assertEqual('dup', branch.name)

    def test_duplicate_name_with_nested_unrelated_name_allowed(self):
        with factory_random.random_branch('a'):
            with factory_random.random_branch('b'):
                with self.assertRaises(RandomBranchError):
                    with factory_random.random_branch('a'):
                        self.fail('outer name must remain reserved')

    def test_auto_names_are_unique(self):
        with factory_random.random_branch() as first:
            with factory_random.random_branch() as second:
                self.assertNotEqual(first.name, second.name)
                self.assertTrue(first.name)
                self.assertTrue(second.name)

    def test_invalid_branch_name(self):
        for bad_name in ('', 0, b'bytes'):
            with self.assertRaises(RandomBranchError):
                factory_random.random_branch(bad_name)

    # -- Exception exits ----------------------------------------------------

    def test_exception_exit_discards_even_after_commit_call(self):
        with self.assertRaises(ValueError):
            with factory_random.random_branch('rollback') as branch:
                [_draw_both() for _ in range(5)]
                branch.commit()
                raise ValueError('kaboom')
        self.assertEqual(self._initial_states, _global_states())
        self.assertIsNone(factory_random.get_active_branch())

    def test_branch_is_usable_again_after_exception(self):
        for attempt in range(3):
            with self.assertRaises(RuntimeError):
                with factory_random.random_branch('reusable'):
                    _fuzzy_value()
                    raise RuntimeError(attempt)
            self.assertEqual(self._initial_states, _global_states())

    def test_commit_or_discard_outside_context_is_an_error(self):
        branch = factory_random.random_branch('standalone')
        with self.assertRaises(RandomBranchError):
            branch.commit()
        with self.assertRaises(RandomBranchError):
            branch.discard()

    def test_conflicting_decisions_are_rejected(self):
        with factory_random.random_branch('conflict') as branch:
            branch.commit()
            with self.assertRaises(RandomBranchError):
                branch.discard()
        with factory_random.random_branch('conflict2') as branch:
            branch.discard()
            with self.assertRaises(RandomBranchError):
                branch.commit()

    # -- Faker integration --------------------------------------------------

    def test_branch_uses_branch_local_faker_instances(self):
        with factory_random.random_branch('faker-local') as branch:
            subfaker = factory.Faker._get_faker()
            self.assertIs(subfaker.random, branch.faker_random)
            self.assertIsNot(subfaker, factory.Faker._FAKER_REGISTRY.get(
                factory.Faker._DEFAULT_LOCALE))
            # Cached per locale within the branch.
            self.assertIs(subfaker, factory.Faker._get_faker())

    def test_custom_provider_is_available_inside_branch(self):
        class EvenProvider(faker.providers.BaseProvider):
            def even(self):
                return self.random_int(0, 100) * 2

        factory.Faker.add_provider(EvenProvider)

        def run():
            with factory_random.random_branch('provider'):
                return [
                    factory.Faker('even').evaluate(
                        None, None, {'locale': None})
                    for _ in range(5)]

        self.assertEqual(run(), run())

    # -- End-to-end with a factory ----------------------------------------

    def test_full_factory_builds_are_repeatable_in_branch(self):
        class Profile:
            def __init__(self, name, score, label):
                self.name = name
                self.score = score
                self.label = label

        class ProfileFactory(factory.Factory):
            class Meta:
                model = Profile

            name = factory.Faker('name')
            score = fuzzy.FuzzyInteger(0, 10 ** 6)
            label = fuzzy.FuzzyChoice(['x', 'y', 'z'])

        def run():
            with factory_random.random_branch('factory-build'):
                return [(p.name, p.score, p.label)
                        for p in ProfileFactory.build_batch(5)]

        first = run()
        factory_random.reseed_random(SEED)
        self.assertEqual(first, run())
        self.assertEqual(self._initial_states, _global_states())

    # -- Thread isolation ---------------------------------------------------

    def test_branches_are_isolated_across_threads(self):
        thread_count = 8
        barrier = threading.Barrier(thread_count + 1, timeout=15)
        results = []
        results_lock = threading.Lock()
        errors = []

        def worker():
            try:
                # Every thread uses the SAME branch name: names are scoped
                # per-thread, so this must not collide.
                with factory_random.random_branch('worker') as branch:
                    fuzzy_start = branch.randgen.getstate()
                    faker_start = branch.faker_random.getstate()
                    barrier.wait()

                    # While workers are inside branches, this thread's branch
                    # must be invisible to the main thread (asserted there);
                    # here, draw and verify against a local replay.
                    fuzzy_values = [_fuzzy_value() for _ in range(50)]
                    faker_values = [_faker_name() for _ in range(3)]

                    fuzzy_replay = random.Random()
                    fuzzy_replay.setstate(fuzzy_start)
                    faker_replay = faker.Faker()
                    faker_replay.random = random.Random()
                    faker_replay.random.setstate(faker_start)

                    ok_fuzzy = fuzzy_values == [
                        fuzzy_replay.randrange(0, 10 ** 9 + 1)
                        for _ in range(50)]
                    ok_faker = faker_values == [
                        faker_replay.name() for _ in range(3)]
                    with results_lock:
                        results.append(ok_fuzzy and ok_faker)
                    branch.discard()
            except Exception as exc:  # pragma: no cover - reported below
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker)
                   for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        barrier.wait()

        # All worker branches are open: the main thread sees none of them,
        # and may itself open a branch bearing the very same name.
        self.assertIsNone(factory_random.get_active_branch())
        with factory_random.random_branch('worker'):
            main_values = [_fuzzy_value() for _ in range(10)]
            self.assertEqual(10, len(main_values))

        for thread in threads:
            thread.join(15)
            self.assertFalse(thread.is_alive())

        self.assertEqual([], errors)
        self.assertEqual([True] * thread_count, results)
        # Every worker discarded: global state is exactly the initial state
        # minus the main thread's own discarded branch (i.e. unchanged).
        self.assertEqual(self._initial_states, _global_states())

    def test_concurrent_branches_never_perturb_global_stream(self):
        """Global draws performed during active branches follow the exact
        stream that would exist without any branch."""
        direct_replay = random.Random()
        direct_replay.setstate(self._initial_states[0])
        faker_replay = faker.Faker()
        faker_replay.random = random.Random()
        faker_replay.random.setstate(self._initial_states[1])

        stop = threading.Event()

        def worker():
            while not stop.is_set():
                with factory_random.random_branch('busy'):
                    [_draw_both() for _ in range(25)]

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()

        expected = []
        observed = []
        for _ in range(200):
            expected.append((
                direct_replay.randrange(0, 10 ** 9 + 1),
                faker_replay.name(),
            ))
            observed.append(_draw_both())

        stop.set()
        for thread in threads:
            thread.join(15)
            self.assertFalse(thread.is_alive())

        self.assertEqual(expected, observed)


if __name__ == '__main__':
    unittest.main()

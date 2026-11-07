# Copyright: See the LICENSE file.

"""Tests for the sequence/random-state transaction snapshot interface."""

import threading
import unittest

import factory
from factory import random as factory_random
from factory.fuzzy import FuzzyInteger
from factory.transactions import (
    SequenceTransaction,
    TransactionResult,
    TransactionState,
    sequence_transaction,
)


class SampleModel:
    """Generic model built by the test factories."""

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


class BasicRollbackTests(unittest.TestCase):

    def _make_factory(self, **declarations):
        attrs = {
            'Meta': type('Meta', (), {'model': SampleModel}),
        }
        attrs.update(declarations)
        return type('F', (factory.Factory,), attrs)

    def test_rollback_restores_sequence(self):
        f = self._make_factory(one=factory.Sequence(lambda n: 'x%d' % n))
        self.assertEqual('x0', f.build().one)

        tx = SequenceTransaction()
        self.assertEqual('x1', f.build().one)
        self.assertEqual('x2', f.build().one)
        tx.rollback()

        # The interval consumed by the transaction is available again.
        self.assertEqual('x1', f.build().one)

    def test_rollback_restores_multi_parameter_sequences(self):
        f = self._make_factory(
            a=factory.Sequence(lambda n: 'a%d' % n),
            b=factory.Sequence(lambda n: 'b%d' % n),
            c=factory.LazyAttributeSequence(lambda obj, n: 'c%d' % n),
        )

        with sequence_transaction() as tx:
            o1 = f.build()
            o2 = f.build()
            self.assertEqual(('a0', 'b0', 'c0'), (o1.a, o1.b, o1.c))
            self.assertEqual(('a1', 'b1', 'c1'), (o2.a, o2.b, o2.c))
            tx.rollback()

        again = f.build()
        self.assertEqual(('a0', 'b0', 'c0'), (again.a, again.b, again.c))

    def test_commit_keeps_consumed_interval(self):
        f = self._make_factory(one=factory.Sequence(lambda n: 'x%d' % n))
        with sequence_transaction():
            self.assertEqual('x0', f.build().one)
            self.assertEqual('x1', f.build().one)
            # Clean exit commits.
        self.assertEqual('x2', f.build().one)

    def test_empty_rollback_is_neutral(self):
        f = self._make_factory(one=factory.Sequence(lambda n: 'x%d' % n))
        f.build()
        f.build()
        with sequence_transaction() as tx:
            tx.rollback()
        # Nothing was consumed inside the transaction; counter unchanged.
        self.assertEqual('x2', f.build().one)

    def test_sequence_override_does_not_consume_counter(self):
        f = self._make_factory(one=factory.Sequence(int))
        f.build()  # seq -> 1

        with sequence_transaction() as tx:
            forced = f.build(__sequence=99)
            self.assertEqual(99, forced.one)
            tx.rollback()

        # The forced sequence must not have advanced the factory counter.
        self.assertEqual(1, f.build().one)

    def test_force_sequence_declarations_roll_back_fine(self):
        # Dict/List declarations reuse the parent sequence; they must not
        # interfere with snapshot restoration.
        f = self._make_factory(
            one=factory.Sequence(lambda n: n),
            items=factory.List([factory.Sequence(lambda n: n)]),
        )
        with sequence_transaction() as tx:
            obj = f.build()
            self.assertEqual(0, obj.one)
            self.assertEqual([0], obj.items)
            tx.rollback()
        obj = f.build()
        self.assertEqual(0, obj.one)
        self.assertEqual([0], obj.items)


class CrossFactoryAndSubFactoryTests(unittest.TestCase):

    def setUp(self):
        class ChildFactory(factory.Factory):
            class Meta:
                model = SampleModel

            one = factory.Sequence(lambda n: 'c%d' % n)

        class ParentFactory(factory.Factory):
            class Meta:
                model = SampleModel

            one = factory.Sequence(lambda n: 'p%d' % n)
            child = factory.SubFactory(ChildFactory)

        class SiblingFactory(factory.Factory):
            class Meta:
                model = SampleModel

            one = factory.Sequence(lambda n: 's%d' % n)

        self.ChildFactory = ChildFactory
        self.ParentFactory = ParentFactory
        self.SiblingFactory = SiblingFactory

    def test_rollback_restores_subfactory_counter(self):
        with sequence_transaction() as tx:
            parent = self.ParentFactory.build()
            self.assertEqual(('p0', 'c0'), (parent.one, parent.child.one))
            tx.rollback()

        again = self.ParentFactory.build()
        self.assertEqual(('p0', 'c0'), (again.one, again.child.one))

    def test_rollback_restores_unrelated_factory_counter_as_well(self):
        # The snapshot spans *all* factories, not only those used in the tx.
        self.ParentFactory.build()  # p1, c1
        self.ParentFactory.build()  # p2, c2

        with sequence_transaction() as tx:
            self.ParentFactory.build()  # p2, c2
            self.SiblingFactory.build()  # s0
            tx.rollback()

        # Parent counters back to where they were before the transaction...
        parent = self.ParentFactory.build()
        self.assertEqual(('p2', 'c2'), (parent.one, parent.child.one))
        # ...and the sibling counter created/used in the tx is reset too.
        self.assertEqual('s0', self.SiblingFactory.build().one)


class NestedTransactionTests(unittest.TestCase):

    def setUp(self):
        class ChildFactory(factory.Factory):
            class Meta:
                model = SampleModel

            one = factory.Sequence(lambda n: 'c%d' % n)

        class ParentFactory(factory.Factory):
            class Meta:
                model = SampleModel

            one = factory.Sequence(lambda n: 'p%d' % n)
            child = factory.SubFactory(ChildFactory)

        self.ChildFactory = ChildFactory
        self.ParentFactory = ParentFactory

    def test_inner_rollback_only_restores_inner_interval(self):
        with sequence_transaction() as outer:
            p0 = self.ParentFactory.build()
            self.assertEqual(('p0', 'c0'), (p0.one, p0.child.one))

            with sequence_transaction() as inner:
                p1 = self.ParentFactory.build()
                p2 = self.ParentFactory.build()
                self.assertEqual(('p1', 'c1'), (p1.one, p1.child.one))
                self.assertEqual(('p2', 'c2'), (p2.one, p2.child.one))
                # Only the [1, 3) interval is rolled back.
                self.assertEqual(TransactionResult.ROLLED_BACK, inner.rollback())

            # The outer transaction's consumption is preserved.
            p_after_inner = self.ParentFactory.build()
            self.assertEqual(
                ('p1', 'c1'),
                (p_after_inner.one, p_after_inner.child.one),
            )

            self.assertEqual(TransactionResult.ROLLED_BACK, outer.rollback())

        # The outer rollback restored the whole range.
        p_after_outer = self.ParentFactory.build()
        self.assertEqual(('p0', 'c0'), (p_after_outer.one, p_after_outer.child.one))

    def test_inner_commit_outer_rollback_restores_all(self):
        with sequence_transaction() as outer:
            self.ParentFactory.build()  # p0, c0
            with sequence_transaction() as inner:
                self.ParentFactory.build()  # p1, c1
                self.ParentFactory.build()  # p2, c2
                self.assertEqual(TransactionResult.COMMITTED, inner.commit())
            self.ParentFactory.build()  # p3, c3
            outer.rollback()

        again = self.ParentFactory.build()
        self.assertEqual(('p0', 'c0'), (again.one, again.child.one))

    def test_inner_rollback_outer_commit_keeps_outer_interval(self):
        with sequence_transaction():
            p0 = self.ParentFactory.build()
            self.assertEqual(('p0', 'c0'), (p0.one, p0.child.one))
            with sequence_transaction() as inner:
                self.ParentFactory.build()  # p1, c1
                inner.rollback()
            # Inner interval is reusable; outer stays active and commits on exit.
            p1 = self.ParentFactory.build()
            self.assertEqual(('p1', 'c1'), (p1.one, p1.child.one))

        p2 = self.ParentFactory.build()
        self.assertEqual(('p2', 'c2'), (p2.one, p2.child.one))

    def test_nested_three_levels_savepoints(self):
        f_child = self.ChildFactory
        with sequence_transaction() as outer:
            self.assertEqual('c0', f_child.build().one)
            with sequence_transaction() as middle:
                self.assertEqual('c1', f_child.build().one)
                with sequence_transaction() as inner:
                    self.assertEqual('c2', f_child.build().one)
                    self.assertEqual('c3', f_child.build().one)
                    inner.rollback()
                self.assertEqual('c2', f_child.build().one)
                middle.rollback()
            self.assertEqual('c1', f_child.build().one)
            outer.rollback()
        self.assertEqual('c0', f_child.build().one)


class ExceptionTests(unittest.TestCase):

    def _make_factory(self):
        return type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            'one': factory.Sequence(lambda n: 'x%d' % n),
        })

    def test_exception_rolls_back_and_propagates(self):
        f = self._make_factory()

        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with sequence_transaction():
                f.build()
                f.build()
                raise Boom()

        self.assertEqual('x0', f.build().one)

    def test_exception_after_manual_commit_does_not_rollback(self):
        f = self._make_factory()

        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with sequence_transaction() as tx:
                f.build()
                f.build()
                tx.commit()
                raise Boom()

        # The explicit commit wins; the exception does not roll it back.
        self.assertEqual('x2', f.build().one)


class TransactionStatusTests(unittest.TestCase):

    def test_states(self):
        tx = SequenceTransaction()
        self.assertIs(TransactionState.ACTIVE, tx.state)
        self.assertEqual(TransactionResult.COMMITTED, tx.commit())
        self.assertIs(TransactionState.COMMITTED, tx.state)

    def test_duplicate_commit_returns_explicit_status(self):
        tx = SequenceTransaction()
        self.assertEqual(TransactionResult.COMMITTED, tx.commit())
        # Illegal duplicate commit is reported, not silently performed.
        self.assertEqual(TransactionResult.ALREADY_COMMITTED, tx.commit())
        self.assertIs(TransactionState.COMMITTED, tx.state)

    def test_duplicate_rollback_returns_explicit_status(self):
        tx = SequenceTransaction()
        self.assertEqual(TransactionResult.ROLLED_BACK, tx.rollback())
        self.assertEqual(TransactionResult.ALREADY_ROLLED_BACK, tx.rollback())
        self.assertIs(TransactionState.ROLLED_BACK, tx.state)

    def test_commit_after_rollback_raises(self):
        tx = SequenceTransaction()
        tx.rollback()
        with self.assertRaises(factory.TransactionError):
            tx.commit()

    def test_rollback_after_commit_raises(self):
        tx = SequenceTransaction()
        tx.commit()
        with self.assertRaises(factory.TransactionError):
            tx.rollback()

    def test_begin_alias(self):
        tx = SequenceTransaction.begin()
        self.assertIs(TransactionState.ACTIVE, tx.state)
        self.assertEqual(TransactionResult.COMMITTED, tx.commit())

    def test_unique_transaction_ids(self):
        tx1 = SequenceTransaction()
        tx2 = SequenceTransaction()
        self.assertNotEqual(tx1.id, tx2.id)
        self.assertIn(str(tx1.id), repr(tx1))
        tx1.rollback()
        tx2.rollback()


class RandomStateTests(unittest.TestCase):

    def test_rollback_restores_random_state(self):
        factory_random.reseed_random(42)
        expected = [factory_random.randgen.randint(0, 10 ** 9) for _ in range(3)]

        factory_random.reseed_random(42)
        with sequence_transaction() as tx:
            inside = [factory_random.randgen.randint(0, 10 ** 9) for _ in range(3)]
            self.assertEqual(expected, inside)
            tx.rollback()

        after = [factory_random.randgen.randint(0, 10 ** 9) for _ in range(3)]
        self.assertEqual(expected, after)

    def test_rollback_restores_state_set_flag(self):
        factory_random.randgen.state_set = False
        with sequence_transaction() as tx:
            factory_random.reseed_random(7)
            self.assertTrue(factory_random.randgen.state_set)
            tx.rollback()
        # The "has the generator ever been seeded" flag is restored too,
        # so transactional rollback cannot change normal fuzzy behaviour.
        self.assertFalse(factory_random.randgen.state_set)

    def test_fuzzy_values_repeat_after_rollback(self):
        f = type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            'n': FuzzyInteger(0, 10 ** 9),
        })
        factory_random.reseed_random(1234)
        with sequence_transaction() as tx:
            first = [f.build().n for _ in range(5)]
            tx.rollback()
        factory_random.reseed_random(1234)
        with sequence_transaction() as tx:
            second = [f.build().n for _ in range(5)]
            tx.rollback()
        self.assertEqual(first, second)


class CounterLifecycleTests(unittest.TestCase):

    def test_counter_created_inside_transaction_is_reset(self):
        f = type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            'one': factory.Sequence(lambda n: 'x%d' % n),
        })
        # The counter is lazily initialized; it does not exist yet.
        self.assertIsNone(f._meta._counter)

        with sequence_transaction() as tx:
            self.assertEqual('x0', f.build().one)
            self.assertEqual('x1', f.build().one)
            tx.rollback()

        # As if the transactional consumption never happened.
        self.assertEqual('x0', f.build().one)

    def test_counter_created_inside_transaction_keeps_setup_value(self):
        f = type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            '_setup_next_sequence': classmethod(lambda cls: 42),
            'one': factory.Sequence(lambda n: 'x%d' % n),
        })
        with sequence_transaction() as tx:
            self.assertEqual('x42', f.build().one)
            tx.rollback()
        self.assertEqual('x42', f.build().one)

    def test_shared_counter_in_inheritance_chain(self):
        class ParentFactory(factory.Factory):
            class Meta:
                model = SampleModel

            one = factory.Sequence(int)

        class ChildFactory(ParentFactory):
            pass

        with sequence_transaction() as tx:
            self.assertEqual(0, ChildFactory.build().one)
            self.assertEqual(1, ParentFactory.build().one)
            tx.rollback()

        self.assertEqual(0, ParentFactory.build().one)
        self.assertEqual(1, ChildFactory.build().one)


class NormalBehaviourTests(unittest.TestCase):

    def test_next_and_reset_unaffected(self):
        f = type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            'one': factory.Sequence(lambda n: 'x%d' % n),
        })

        self.assertEqual('x0', f.build().one)
        self.assertEqual('x1', f.build().one)

        f.reset_sequence()
        self.assertEqual('x0', f.build().one)

        f.reset_sequence(5)
        self.assertEqual('x5', f.build().one)
        self.assertEqual('x6', f.build().one)

        # Rollback after the fact restores the value captured at begin.
        with sequence_transaction() as tx:
            f.build()
            f.build()
            tx.rollback()
        self.assertEqual('x7', f.build().one)

    def test_reset_sequence_inside_transaction_is_rolled_back(self):
        f = type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            'one': factory.Sequence(int),
        })
        f.build()
        f.build()  # counter -> 2
        with sequence_transaction() as tx:
            f.reset_sequence(100)
            self.assertEqual(100, f.build().one)
            tx.rollback()
        self.assertEqual(2, f.build().one)


class ConcurrencyTests(unittest.TestCase):
    """Transactions must be serialized so they never overwrite each other."""

    WORKERS = 8

    def _make_factory(self):
        return type('F', (factory.Factory,), {
            'Meta': type('Meta', (), {'model': SampleModel}),
            'one': factory.Sequence(int),
        })

    def _run_workers(self, target, timeout=10):
        errors = []
        threads = []
        barrier = threading.Barrier(self.WORKERS)

        def wrapped():
            barrier.wait()
            try:
                target()
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        for _ in range(self.WORKERS):
            thread = threading.Thread(target=wrapped)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout)
            self.assertFalse(
                thread.is_alive(),
                'transaction worker hung (deadlock?)',
            )
        self.assertEqual([], errors)

    def test_concurrent_rollback_transactions_are_isolated(self):
        f = self._make_factory()

        def worker():
            with sequence_transaction() as tx:
                values = [f.build().one for _ in range(2)]
                self.assertEqual([0, 1], values)
                tx.rollback()

        self._run_workers(worker)
        # Every transaction restored the counter; no worker overwrote another.
        self.assertEqual(0, f.build().one)

    def test_concurrent_commit_transactions_consume_disjoint_intervals(self):
        f = self._make_factory()
        seen = []
        seen_lock = threading.Lock()

        def worker():
            with sequence_transaction():
                values = [f.build().one for _ in range(2)]
                with seen_lock:
                    seen.extend(values)
                # Clean exit commits.

        self._run_workers(worker)
        # Serialized commits produce exactly one contiguous, disjoint set.
        self.assertEqual(list(range(2 * self.WORKERS)), sorted(seen))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(2 * self.WORKERS, f.build().one)

    def test_concurrent_duplicate_commit_has_single_winner(self):
        tx = SequenceTransaction()
        barrier = threading.Barrier(2)
        results = []
        results_lock = threading.Lock()

        def other_thread():
            barrier.wait()
            result = tx.commit()
            with results_lock:
                results.append(result)

        thread = threading.Thread(target=other_thread)
        thread.start()
        barrier.wait()
        try:
            results.append(tx.commit())
        finally:
            thread.join(10)
        self.assertFalse(thread.is_alive())

        # Exactly one commit succeeded; the duplicate was reported explicitly.
        self.assertEqual(
            {TransactionResult.COMMITTED, TransactionResult.ALREADY_COMMITTED},
            set(results),
        )


if __name__ == '__main__':
    unittest.main()

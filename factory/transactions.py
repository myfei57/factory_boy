# Copyright: See the LICENSE file.

"""Transactional snapshots of sequence counters and random state.

A :class:`SequenceTransaction` takes a snapshot of every Factory sequence
counter (including those used by nested ``SubFactory`` calls) and of the
factory_boy random generator right before a group of related Factory calls.
It can then either commit (keep the consumed sequence values and random
consumption) or roll back (restore the captured snapshot).

Transactions may be nested; inner transactions behave like database
savepoints: rolling them back restores only the sequence interval consumed
inside the inner transaction.

Transactions are serialized across threads, so concurrent transactions can
never overwrite each other's state. Non-transactional sequence use
(``next``/``reset_sequence``) is completely unaffected and never acquires any
transaction lock.
"""

import enum
import itertools
import threading
import typing as T
import weakref

import faker.generator

from . import random as random_mod
from .errors import TransactionError

if T.TYPE_CHECKING:
    from .base import _Counter


class TransactionState(enum.Enum):
    """Lifecycle state of a :class:`SequenceTransaction`."""

    #: The transaction is open; commit() or rollback() may still be called.
    ACTIVE = 'active'
    #: The transaction has been committed.
    COMMITTED = 'committed'
    #: The transaction has been rolled back; the snapshot was restored.
    ROLLED_BACK = 'rolled_back'


class TransactionResult(enum.Enum):
    """Result of a :meth:`SequenceTransaction.commit`/`rollback` request.

    The ``ALREADY_*`` values give an explicit status to illegal duplicate
    requests, instead of silently pretending the request was performed again.
    """

    #: The transaction was committed by this call.
    COMMITTED = 'committed'
    #: The transaction was rolled back by this call.
    ROLLED_BACK = 'rolled_back'
    #: commit() was called on an already-committed transaction.
    ALREADY_COMMITTED = 'already_committed'
    #: rollback() was called on an already-rolled-back transaction.
    ALREADY_ROLLED_BACK = 'already_rolled_back'


#: Guards all shared transaction bookkeeping (counter registry, snapshots).
_registry_lock = threading.RLock()

#: Serializes whole transactions across threads; reentrant so that a
#: transaction opened within another one (in the same thread) nests.
_global_lock = threading.RLock()

#: Maps every live sequence counter to its initial (post-construction,
#: pre-consumption) sequence value. Weak references avoid retaining counters
#: (and their factory classes) beyond their natural lifetime.
_counter_initial: 'weakref.WeakKeyDictionary[_Counter, int]' = (
    weakref.WeakKeyDictionary()
)

#: Source of unique transaction identifiers.
_transaction_ids = itertools.count(1)


def _register_counter(counter, initial_seq):
    """Record a newly created sequence counter.

    Called from ``_Counter.__init__``; this never alters counter behaviour.
    """
    with _registry_lock:
        _counter_initial[counter] = initial_seq


def _capture_random_state():
    """Snapshot the fuzzy/Faker random generator state.

    Both the random state and the ``state_set`` flag are captured, so that
    rolling back restores the exact observable state (including whether the
    user ever explicitly seeded the generator).
    """
    return (
        random_mod.randgen.getstate(),
        random_mod.randgen.state_set,
    )


def _restore_random_state(snapshot):
    """Restore a snapshot taken by :func:`_capture_random_state`."""
    state, state_set = snapshot
    random_mod.randgen.setstate(state)
    random_mod.randgen.state_set = state_set
    faker.generator.random.setstate(state)


class SequenceTransaction:
    """Snapshot sequence counters and random state, then commit or roll back.

    Usage (context manager, recommended)::

        with sequence_transaction():
            MyFactory()
            OtherFactory()  # including nested SubFactory calls

    The block commits on clean exit; if an exception escapes the block, the
    transaction is rolled back (every counter and the random generator are
    restored to their state at transaction start) and the exception is
    re-raised.

    Explicit usage::

        tx = SequenceTransaction()
        try:
            MyFactory()
            tx.commit()
        except Exception:
            tx.rollback()
            raise

    Nested transactions act as savepoints: an inner rollback restores only
    the interval consumed within the inner transaction, leaving the outer
    transaction's consumption untouched.

    Transactions are serialized across threads: constructing a transaction
    blocks while another thread has an open transaction, which guarantees
    that concurrent transactions never overwrite each other. Normal,
    non-transactional Factory calls are not serialized.
    """

    def __init__(self):
        self._id = next(_transaction_ids)
        self._state = TransactionState.ACTIVE
        self._owner = threading.get_ident()
        self._locks_held = 0
        self._counter_snapshot = {}
        self._random_snapshot = None

        # Serialize against transactions opened by other threads *before*
        # taking the snapshot, so the snapshot cannot contain another
        # transaction's consumption.
        self._acquire_global_lock()
        try:
            self._capture_snapshot()
        except BaseException:
            self._release_global_lock()
            raise

    # -- Construction helpers ------------------------------------------------

    @classmethod
    def begin(cls):
        """Begin a new transaction.

        This is an explicit, readable alias for ``SequenceTransaction()``.
        """
        return cls()

    # -- Lock management -----------------------------------------------------

    def _acquire_global_lock(self):
        _global_lock.acquire()
        self._locks_held += 1

    def _release_global_lock(self):
        """Release every global-lock depth held by this transaction.

        Only the owning thread releases the lock; this makes finalization
        safe when commit()/rollback() happens to be probed from another
        thread (the owner remains responsible for the eventual release).
        """
        if self._owner != threading.get_ident():
            return
        while self._locks_held:
            _global_lock.release()
            self._locks_held -= 1

    # -- Snapshot management -------------------------------------------------

    def _capture_snapshot(self):
        with _registry_lock:
            self._counter_snapshot = {
                counter: counter.seq
                for counter in list(_counter_initial)
            }
            self._random_snapshot = _capture_random_state()

    def _restore_snapshot(self):
        with _registry_lock:
            for counter, initial_seq in list(_counter_initial.items()):
                if counter in self._counter_snapshot:
                    # Counter existed when the transaction began: restore the
                    # exact value captured then.
                    counter.reset(self._counter_snapshot[counter])
                else:
                    # The counter was lazily created during the transaction:
                    # restore its fresh, pre-consumption value.
                    counter.reset(initial_seq)
            _restore_random_state(self._random_snapshot)

    # -- Public API ----------------------------------------------------------

    @property
    def id(self):
        """Unique identifier of this transaction."""
        return self._id

    @property
    def state(self):
        """Current :class:`TransactionState` of the transaction."""
        return self._state

    def commit(self):
        """Commit the transaction, keeping all consumed sequence values.

        Returns:
            TransactionResult.COMMITTED on the first, valid call;
            TransactionResult.ALREADY_COMMITTED if the transaction was
            already committed.

        Raises:
            TransactionError: if the transaction has already been rolled back.
        """
        with _registry_lock:
            if self._state is TransactionState.COMMITTED:
                result = TransactionResult.ALREADY_COMMITTED
            elif self._state is TransactionState.ROLLED_BACK:
                raise TransactionError(
                    "Cannot commit transaction #%d: it has already been "
                    "rolled back." % self._id)
            else:
                self._state = TransactionState.COMMITTED
                result = TransactionResult.COMMITTED
        self._release_global_lock()
        return result

    def rollback(self):
        """Roll the transaction back, restoring the captured snapshot.

        Returns:
            TransactionResult.ROLLED_BACK on the first, valid call;
            TransactionResult.ALREADY_ROLLED_BACK if the transaction was
            already rolled back.

        Raises:
            TransactionError: if the transaction has already been committed.
        """
        with _registry_lock:
            if self._state is TransactionState.ROLLED_BACK:
                result = TransactionResult.ALREADY_ROLLED_BACK
            elif self._state is TransactionState.COMMITTED:
                raise TransactionError(
                    "Cannot roll back transaction #%d: it has already been "
                    "committed." % self._id)
            else:
                self._restore_snapshot()
                self._state = TransactionState.ROLLED_BACK
                result = TransactionResult.ROLLED_BACK
        self._release_global_lock()
        return result

    # -- Context manager -----------------------------------------------------

    def __enter__(self):
        # Reentrant: nesting transactions in the same thread adds one lock
        # depth, released when this transaction is finalized.
        self._acquire_global_lock()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                if self._state is TransactionState.ACTIVE:
                    self.commit()
            elif self._state is TransactionState.ACTIVE:
                self.rollback()
        finally:
            # Safety net: always release on exit, even if the user already
            # finalized the transaction inside the block.
            self._release_global_lock()
        # Never swallow the exception.
        return False

    def __repr__(self):
        return '<SequenceTransaction #%d: %s>' % (self._id, self._state.value)


def sequence_transaction():
    """Begin a transactional snapshot of sequence counters and random state.

    Typical usage::

        with sequence_transaction():
            MyFactory()
            OtherFactory()

    See :class:`SequenceTransaction` for details.
    """
    return SequenceTransaction()

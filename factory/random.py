import itertools
import random
import threading

import faker.generator

from .errors import RandomBranchError

randgen = random.Random()

randgen.state_set = False


def get_random_state():
    """Retrieve the state of factory.fuzzy's random generator."""
    state = randgen.getstate()
    # Returned state must represent both Faker and factory_boy.
    faker.generator.random.setstate(state)
    return state


def set_random_state(state):
    """Force-set the state of factory.fuzzy's random generator."""
    randgen.state_set = True
    randgen.setstate(state)

    faker.generator.random.setstate(state)


def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = random.Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)


# ---------------------------------------------------------------------------
# Random state branches
# ---------------------------------------------------------------------------

_COMMIT = 'commit'
_DISCARD = 'discard'

_branch_name_counter = itertools.count(1)

# Each thread owns an independent stack of active branches; branch state is
# therefore never shared across threads.
_thread_local = threading.local()


def _get_stack():
    stack = getattr(_thread_local, 'stack', None)
    if stack is None:
        stack = []
        _thread_local.stack = stack
    return stack


def get_active_branch():
    """Return the currently-active :class:`RandomBranch` in this thread.

    Returns ``None`` when no :func:`random_branch` context is active in the
    current thread.
    """
    stack = getattr(_thread_local, 'stack', None)
    if stack:
        return stack[-1]
    return None


def get_active_random():
    """Return the :class:`random.Random` used by ``factory.fuzzy`` right now.

    Inside a :func:`random_branch` context this is the branch-local generator;
    outside of one it is the global :data:`randgen`.
    """
    branch = get_active_branch()
    if branch is None:
        return randgen
    return branch.randgen


class RandomBranch:
    """A fork of the factory/faker random state, usable as a context manager.

    Instances are created through :func:`random_branch`. Inside the context,
    :mod:`factory.fuzzy` declarations and :class:`factory.Faker` declarations
    draw from private copies of the parent random generators, so results are
    reproducible and do not perturb the parent state until :meth:`commit` is
    called.

    On exit:

    * :meth:`commit` (only honored on a clean exit) advances the parent
      generators to the branch's end state;
    * :meth:`discard`, an exception propagating through the ``with`` block,
      or simply exiting without choosing restores the parent state untouched.

    Branches may be nested; each thread has its own branch stack.
    """

    def __init__(self, name=None):
        if name is None:
            name = 'branch-%d' % next(_branch_name_counter)
        if not isinstance(name, str) or not name:
            raise RandomBranchError(
                "A random branch name must be a non-empty string, got %r."
                % (name,))
        self.name = name
        self._parent = None
        self._randgen = None
        self._faker_random = None
        self._fakers = {}
        self._decision = None
        self._active = False

    # -- Introspection -----------------------------------------------------

    @property
    def randgen(self):
        """The :class:`random.Random` backing ``factory.fuzzy`` in-branch."""
        return self._randgen

    @property
    def faker_random(self):
        """The :class:`random.Random` backing ``factory.Faker`` in-branch."""
        return self._faker_random

    @property
    def parent(self):
        """The branch this one was forked from, or ``None`` at the root."""
        return self._parent

    @property
    def active(self):
        """Whether the branch context is currently entered in this thread."""
        return self._active

    @property
    def committed(self):
        return self._decision == _COMMIT

    @property
    def discarded(self):
        return self._decision == _DISCARD

    def get_state(self):
        """Return the current ``(fuzzy_state, faker_state)`` of the branch."""
        return self._randgen.getstate(), self._faker_random.getstate()

    # -- Context management ------------------------------------------------

    def __enter__(self):
        stack = _get_stack()
        for active_branch in stack:
            if active_branch.name == self.name:
                raise RandomBranchError(
                    "A random branch named %r is already active in this "
                    "thread; branch names must be unique within a thread."
                    % self.name)

        self._parent = stack[-1] if stack else None
        if self._parent is None:
            parent_randgen = randgen
            parent_faker_random = faker.generator.random
        else:
            parent_randgen = self._parent._randgen
            parent_faker_random = self._parent._faker_random

        # Fork: fresh generators seeded with copies of the parent states.
        self._randgen = random.Random()
        self._randgen.setstate(parent_randgen.getstate())
        self._randgen.state_set = getattr(parent_randgen, 'state_set', False)

        self._faker_random = random.Random()
        self._faker_random.setstate(parent_faker_random.getstate())
        self._faker_random.state_set = getattr(
            parent_faker_random, 'state_set', False)

        stack.append(self)
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack = _get_stack()
        try:
            for position in range(len(stack) - 1, -1, -1):
                if stack[position] is self:
                    del stack[position]
                    break
        finally:
            self._active = False

        # Commit only when the block completed cleanly and asked for it.
        # Any exception discards the branch, even after commit() was called.
        if exc_type is None and self._decision == _COMMIT:
            self._apply_to_parent()
        return False

    # -- Decisions ---------------------------------------------------------

    def commit(self):
        """Mark the branch for merge into its parent on clean exit."""
        if not self._active:
            raise RandomBranchError(
                "Cannot commit random branch %r outside of its context."
                % self.name)
        if self._decision == _DISCARD:
            raise RandomBranchError(
                "Random branch %r was already marked for discard."
                % self.name)
        self._decision = _COMMIT
        return self

    def discard(self):
        """Mark the branch to be thrown away on exit (the default)."""
        if not self._active:
            raise RandomBranchError(
                "Cannot discard random branch %r outside of its context."
                % self.name)
        if self._decision == _COMMIT:
            raise RandomBranchError(
                "Random branch %r was already marked for commit."
                % self.name)
        self._decision = _DISCARD
        return self

    def _apply_to_parent(self):
        if self._parent is None:
            target_randgen = randgen
            target_faker_random = faker.generator.random
        else:
            target_randgen = self._parent._randgen
            target_faker_random = self._parent._faker_random
        target_randgen.setstate(self._randgen.getstate())
        target_faker_random.setstate(self._faker_random.getstate())

    # -- Faker integration -------------------------------------------------

    def get_faker(self, locale, creator):
        """Return the branch-local faker for ``locale``.

        ``creator`` is a zero-argument callable building a fresh faker bound
        to this branch's faker random generator; it is invoked at most once
        per locale.
        """
        if locale not in self._fakers:
            self._fakers[locale] = creator()
        return self._fakers[locale]


def random_branch(name=None):
    """Fork the current factory/faker random state into a named branch.

    Args:
        name (str): optional branch name; must be unique among the branches
            currently active in this thread. An auto-generated name is used
            when omitted.

    Returns:
        RandomBranch: a context manager. Draws made through
        :mod:`factory.fuzzy` and :class:`factory.Faker` inside the ``with``
        block consume the forked state; they do not touch the parent (or
        global) generators until :meth:`~RandomBranch.commit` is called on a
        clean exit.

    Example::

        with factory.random.random_branch('fixtures') as branch:
            value = factory.fuzzy.FuzzyInteger(0, 100).fuzz()
            branch.commit()  # propagate consumed randomness to the parent
    """
    return RandomBranch(name=name)

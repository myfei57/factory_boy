# Copyright: See the LICENSE file.


"""Aggregation of post-generation declaration execution.

This module collects, for every post-generation declaration (such as
:class:`~factory.PostGeneration` hooks or
:class:`~factory.RelatedFactory`) executed while building an object:

- whether it was *skipped*, *executed* or *failed*,
- the value it returned,
- the exception raised when it failed,

and merges them into an immutable :class:`PostGenerationReport` ordered as the
declarations were declared.

The aggregator never re-runs a declaration: each declaration is evaluated
exactly once, and a failure either stops the evaluation (keeping the boundary
of declarations already executed) or, at the caller's request, lets the
remaining declarations run.

Collecting reports does not change the return value of
:meth:`factory.Factory.build` / :meth:`factory.Factory.create`; use the
:func:`collect_postgeneration_reports` context manager to access reports.
"""

import contextlib
import contextvars
import dataclasses
import typing as T
from types import MappingProxyType

from . import declarations, enums, errors


class PostGenerationStatus:
    """Possible statuses of a post-generation declaration."""

    #: The declaration was not executed (it evaluated to the SKIP marker).
    SKIPPED = 'skipped'

    #: The declaration was executed and returned a value.
    EXECUTED = 'executed'

    #: The declaration was executed but raised an exception.
    FAILED = 'failed'

    ALL = frozenset((SKIPPED, EXECUTED, FAILED))


@dataclasses.dataclass(frozen=True)
class PostGenerationResult:
    """The outcome of a single post-generation declaration.

    Attributes:
        index (int): position of the declaration, in declaration order.
        name (str): name of the declaration on the factory.
        status (str): one of :class:`PostGenerationStatus` values.
        value: value returned by the declaration; only set when the
            declaration was executed.
        exception: exception raised by the declaration, if any.
    """

    index: int
    name: str
    status: str
    value: T.Any = None
    exception: T.Optional[BaseException] = None

    @property
    def was_executed(self):
        return self.status == PostGenerationStatus.EXECUTED

    @property
    def was_skipped(self):
        return self.status == PostGenerationStatus.SKIPPED

    @property
    def failed(self):
        return self.status == PostGenerationStatus.FAILED

    def __str__(self):
        if self.status == PostGenerationStatus.FAILED:
            return f'#{self.index} {self.name}: {self.status} ({self.exception!r})'
        return f'#{self.index} {self.name}: {self.status}'


@dataclasses.dataclass(frozen=True)
class PostGenerationReport:
    """Immutable, declaration-ordered report of post-generation execution.

    Attributes:
        entries (tuple of PostGenerationResult): the recorded results, in
            declaration order.
        interrupted (bool): True when a failure stopped evaluation before all
            declarations could run; declarations past ``entries`` were never
            executed.
    """

    entries: T.Tuple[PostGenerationResult, ...] = ()
    interrupted: bool = False

    #: Shared empty report.
    EMPTY: T.ClassVar['PostGenerationReport']

    def __post_init__(self):
        # MappingProxyType gives an immutable, read-only view.
        by_name = MappingProxyType({entry.name: entry for entry in self.entries})
        return_values = MappingProxyType({
            entry.name: entry.value
            for entry in self.entries
            if entry.status == PostGenerationStatus.EXECUTED
        })
        object.__setattr__(self, '_by_name', by_name)
        object.__setattr__(self, '_return_values', return_values)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, name):
        return name in self._by_name

    def __getitem__(self, key):
        """Retrieve a result by declaration name or positional index."""
        if isinstance(key, int):
            return self.entries[key]
        return self._by_name[key]

    def __eq__(self, other):
        if isinstance(other, PostGenerationReport):
            return (
                self.entries == other.entries
                and self.interrupted == other.interrupted
            )
        return NotImplemented

    @property
    def names(self):
        """Names of the recorded declarations, in declaration order."""
        return tuple(entry.name for entry in self.entries)

    @property
    def executed(self):
        """Results of declarations that executed successfully."""
        return tuple(
            entry for entry in self.entries
            if entry.status == PostGenerationStatus.EXECUTED
        )

    @property
    def skipped(self):
        """Results of declarations that were skipped."""
        return tuple(
            entry for entry in self.entries
            if entry.status == PostGenerationStatus.SKIPPED
        )

    @property
    def failed(self):
        """Results of declarations that raised an exception."""
        return tuple(
            entry for entry in self.entries
            if entry.status == PostGenerationStatus.FAILED
        )

    @property
    def return_values(self):
        """Immutable mapping of declaration name to its return value.

        Only executed declarations are present.
        """
        return self._return_values

    @property
    def ok(self):
        """True if no declaration failed."""
        return not any(
            entry.status == PostGenerationStatus.FAILED
            for entry in self.entries
        )

    def raise_if_failed(self):
        """Raise a :class:`~factory.errors.PostGenerationError` on failure.

        The raised error carries this report (keeping executed-declaration
        boundaries) and is chained on the first recorded exception; does
        nothing when the report is successful.
        """
        for entry in self.entries:
            if entry.status == PostGenerationStatus.FAILED:
                raise errors.PostGenerationError(
                    report=self,
                    declaration=entry.name,
                    exception=entry.exception,
                ) from entry.exception


PostGenerationReport.EMPTY = PostGenerationReport()


class PostGenerationOutcome(T.NamedTuple):
    """Result of evaluating a set of post-generation declarations.

    Attributes:
        results (dict): declaration name mapped to the declaration's raw
            return value, in the same shape historically passed to
            ``_after_postgeneration``.
        report (PostGenerationReport): the immutable aggregated report.
    """

    results: T.Dict[str, T.Any]
    report: PostGenerationReport


class PostGenerationAggregator:
    """Evaluate post-generation declarations once, aggregating their results.

    Args:
        on_error (str): :attr:`STOP` (the default) halts evaluation at the
            first failure, preserving the boundary of already-executed
            declarations; :attr:`CONTINUE` keeps evaluating the remaining
            declarations, recording every failure.
    """

    #: Stop evaluating further declarations as soon as one fails.
    STOP = 'stop'

    #: Keep evaluating remaining declarations after a failure.
    CONTINUE = 'continue'

    POLICIES = frozenset((STOP, CONTINUE))

    def __init__(self, on_error: str = STOP):
        if on_error not in self.POLICIES:
            raise ValueError(
                "Invalid post-generation error policy %r; use %r or %r."
                % (on_error, self.STOP, self.CONTINUE)
            )
        self.on_error = on_error

    def aggregate(self, post_declarations, instance, step):
        """Evaluate every declaration once and merge their results.

        Args:
            post_declarations (factory.builder.DeclarationSet): the
                post-generation declarations to run, as produced by
                :func:`factory.builder.parse_declarations`.
            instance (object): the freshly generated object.
            step (factory.builder.BuildStep): the current build step.

        Returns:
            PostGenerationOutcome: the legacy results dict plus the immutable
            report.
        """
        results = {}
        entries = []
        interrupted = False

        ordered_names = post_declarations.sorted()
        for index, declaration_name in enumerate(ordered_names):
            declaration = post_declarations[declaration_name]
            try:
                raw_value = declaration.declaration.evaluate_post(
                    instance=instance,
                    step=step,
                    overrides=declaration.context,
                )
            except Exception as exc:  # noqa: BLE001 - record the failure
                entries.append(PostGenerationResult(
                    index=index,
                    name=declaration_name,
                    status=PostGenerationStatus.FAILED,
                    exception=exc,
                ))
                if self.on_error == self.STOP:
                    interrupted = True
                    break
                continue

            # Keep the legacy results mapping byte-for-byte identical to the
            # historical build loop (including a SKIP marker).
            results[declaration_name] = raw_value

            if isinstance(raw_value, declarations.Skip):
                entries.append(PostGenerationResult(
                    index=index,
                    name=declaration_name,
                    status=PostGenerationStatus.SKIPPED,
                ))
            else:
                entries.append(PostGenerationResult(
                    index=index,
                    name=declaration_name,
                    status=PostGenerationStatus.EXECUTED,
                    value=raw_value,
                ))

        report = PostGenerationReport(
            entries=tuple(entries),
            interrupted=interrupted,
        )
        return PostGenerationOutcome(results=results, report=report)

    def run(self, post_declarations, instance, step):
        """Return only the :class:`PostGenerationReport` from :meth:`aggregate`."""
        return self.aggregate(post_declarations, instance, step).report


def evaluate_post_declarations(post_declarations, instance, step, *, on_error=PostGenerationAggregator.STOP):
    """Evaluate post-generation declarations through a fresh aggregator.

    Each declaration is evaluated exactly once; no hook is ever re-executed.
    """
    return PostGenerationAggregator(on_error=on_error).aggregate(
        post_declarations, instance, step,
    )


# Report collection (side-channel that leaves build/create return values alone)
# =============================================================================


class CollectedPostGeneration(T.NamedTuple):
    """A report collected while building an object.

    Attributes:
        factory: the factory class being built.
        instance (object): the object that was built.
        create (bool): whether the build used the create strategy.
        report (PostGenerationReport): the aggregated report.
    """

    factory: type
    instance: T.Any
    create: bool
    report: PostGenerationReport


class PostGenerationReportCollector:
    """Sink for post-generation reports produced during ``build``/``create``.

    Instances are created by :func:`collect_postgeneration_reports`; they
    cannot start a second evaluation of any hook, they only record reports
    from the single evaluation performed while building.
    """

    def __init__(self, *, on_error=PostGenerationAggregator.STOP):
        if on_error not in PostGenerationAggregator.POLICIES:
            raise ValueError(
                "Invalid post-generation error policy %r; use %r or %r."
                % (on_error, PostGenerationAggregator.STOP,
                   PostGenerationAggregator.CONTINUE)
            )
        self.on_error = on_error
        self._records = []

    @property
    def records(self):
        """Collected records, in completion order (nested builds first)."""
        return tuple(self._records)

    @property
    def reports(self):
        """Collected reports, in completion order."""
        return tuple(record.report for record in self._records)

    @property
    def report(self):
        """Report of the outermost (last-completed) build.

        Returns an empty report when nothing was collected.
        """
        if self._records:
            return self._records[-1].report
        return PostGenerationReport.EMPTY

    def for_factory(self, factory):
        """Reports collected while building the given factory class."""
        return tuple(
            record.report
            for record in self._records
            if record.factory is factory
        )

    def _record_build(self, factory_meta, step, instance, report):
        self._records.append(CollectedPostGeneration(
            factory=factory_meta.factory,
            instance=instance,
            create=step.builder.strategy == enums.CREATE_STRATEGY,
            report=report,
        ))


_current_collector = contextvars.ContextVar(
    'factory_postgeneration_collector', default=None,
)


def get_current_collector():
    """Return the active collector, or None."""
    return _current_collector.get()


@contextlib.contextmanager
def collect_postgeneration_reports(*, on_error=PostGenerationAggregator.STOP):
    """Collect post-generation reports from builds inside the ``with`` block.

    Example::

        with collect_postgeneration_reports() as collected:
            obj = MyFactory.build()
        report = collected.report
        [entry.status for entry in report]  # ['executed', ...]

    Args:
        on_error (str): :attr:`PostGenerationAggregator.STOP` keeps the
            historical behaviour (the original exception propagates and later
            declarations do not run);
            :attr:`PostGenerationAggregator.CONTINUE` records failures in the
            report and lets the remaining declarations run, leaving the
            build/create return value untouched.

    Yields:
        PostGenerationReportCollector: the collector receiving reports from
        the enclosing build as well as nested ones (e.g. RelatedFactory).
    """
    collector = PostGenerationReportCollector(on_error=on_error)
    token = _current_collector.set(collector)
    try:
        yield collector
    finally:
        _current_collector.reset(token)

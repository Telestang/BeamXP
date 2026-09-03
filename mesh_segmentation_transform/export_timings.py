"""Where an export's wall clock actually goes, reported to the console.

The per-texture ``phase_timings`` already in each plan answer "what did this
job do"; they cannot answer "what is this export spending its time on", for
three reasons.  They only cover the correction pass, so the detect-only
pre-pass -- which on the ardente is where every detector call really happens,
the correction pass reusing the session cache -- is invisible. They stop at the
phase boundary, so a phase that is 60% one function reads as one number. And
they never leave the worker that produced them.

This is the other half: one ledger per process, merged back from the pool, and
printed as a table at the end of an export.

2026-09-02: added after an optimisation question could not be answered without
a throwaway script. The first run of it was worth having -- 48% of a serial
ardente export was in nothing any existing timer named.

Deliberately not a profiler. It measures a dozen named things that are worth
knowing about and costs two ``perf_counter`` calls each; a real profiler
answers a different and much more expensive question.
"""

from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

# The order components are reported in when their cost ties, and the set the
# summary knows how to group.  A label absent from here still records and still
# prints; it just sorts by cost alone.
DETECTION = "detection"
GEOMETRY = "mesh geometry"
PLANNING = "flip planning"
APPLYING = "applying flips"
ENCODING = "texture encoding"
IO = "image and archive I/O"

GROUPS: dict[str, str] = {
    "detect (glyph_lab)": DETECTION,
    "detect (legacy)": DETECTION,
    "normal evidence": DETECTION,
    "repeat-texture index": DETECTION,
    "build domain masks": GEOMETRY,
    "mesh seams": GEOMETRY,
    "uv islands": GEOMETRY,
    "symmetry topology": GEOMETRY,
    "symmetry preparation": GEOMETRY,
    "symmetry segmentation": GEOMETRY,
    "symmetry boundary filters": GEOMETRY,
    "symmetry perimeter fit": GEOMETRY,
    "symmetry child adoption": GEOMETRY,
    "symmetry recovery": GEOMETRY,
    "resolve bindings": GEOMETRY,
    "plan island flips": PLANNING,
    "plan simple uv flips": PLANNING,
    "skew frames": PLANNING,
    "skew per region": PLANNING,
    "exchangeable share": PLANNING,
    "dedupe regions": PLANNING,
    "resolve symbols": GEOMETRY,
    "detection cache key": DETECTION,
    "apply flip plan": APPLYING,
    "encode dds": ENCODING,
    "write png": ENCODING,
    "load texture": IO,
    "extract archive member": IO,
}


@dataclass(slots=True)
class TimingLedger:
    """Seconds and call counts per named component, safe across threads."""

    seconds: Counter = field(default_factory=Counter)
    calls: Counter = field(default_factory=Counter)
    # How many spans were open at once.  Above one means the sum of the
    # components is CPU time spread over threads, not a share of the wall.
    peak_concurrency: int = 0
    _open: int = field(default=0, repr=False, compare=False)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @contextmanager
    def span(self, label: str):
        started = time.perf_counter()
        with self._lock:
            self._open += 1
            self.peak_concurrency = max(self.peak_concurrency, self._open)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._open -= 1
                self.seconds[label] += elapsed
                self.calls[label] += 1

    def merge(self, other: TimingLedger) -> None:
        with self._lock:
            self.seconds.update(other.seconds)
            self.calls.update(other.calls)

    def snapshot(self) -> tuple[Counter, Counter]:
        """A picklable copy, for returning from a worker process."""
        with self._lock:
            return Counter(self.seconds), Counter(self.calls)

    def absorb(self, snapshot: tuple[Counter, Counter] | None) -> None:
        if not snapshot:
            return
        seconds, calls = snapshot
        with self._lock:
            self.seconds.update(seconds)
            self.calls.update(calls)

    def clear(self) -> None:
        with self._lock:
            self.seconds.clear()
            self.calls.clear()

    def summary(self, wall_seconds: float, workers: int = 1) -> list[str]:
        """The report, as log lines.

        ``workers`` is how many processes the measured work was spread over.
        With a pool the components sum to more than the wall clock, which is
        the point of having run one -- so the share column is against the total
        measured rather than against wall, and the wall line says both.
        """
        with self._lock:
            seconds = Counter(self.seconds)
            calls = Counter(self.calls)
        measured = sum(seconds.values())
        if not measured:
            return []
        lines = [
            f"  timing: {wall_seconds:.1f}s wall"
            + (f" over {workers} worker(s)" if workers > 1 else "")
            + f"; {measured:.1f}s measured across {sum(calls.values())} call(s)"
        ]
        grouped: Counter = Counter()
        for label, value in seconds.items():
            grouped[GROUPS.get(label, "other")] += value
        for group, group_seconds in grouped.most_common():
            lines.append(
                f"    {group:<22} {group_seconds:7.1f}s {group_seconds / measured:6.1%}"
            )
            members = sorted(
                ((label, value) for label, value in seconds.items()
                 if GROUPS.get(label, "other") == group),
                key=lambda item: item[1],
                reverse=True,
            )
            for label, value in members:
                lines.append(
                    f"      {label:<20} {value:7.1f}s {value / measured:6.1%}"
                    f"  x{calls[label]}"
                )
        unmeasured = wall_seconds - measured / max(workers, 1)
        if workers == 1 and unmeasured > 0.05 * wall_seconds:
            lines.append(
                f"    {'unmeasured':<22} {unmeasured:7.1f}s "
                f"{unmeasured / wall_seconds:6.1%} of wall"
            )
        return lines


# One ledger per process.  Workers get their own by virtue of being separate
# processes and hand it back through their outcome; nothing here is shared
# across a process boundary while it is being written.
TIMINGS = TimingLedger()


@contextmanager
def timed(label: str):
    """Record one span against this process's ledger."""
    with TIMINGS.span(label):
        yield

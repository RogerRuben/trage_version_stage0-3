"""Immutable session windows plus a small, monotone runtime event calendar.

Only exogenous admission windows are indexed. Busy/free status and positions
are ALWAYS read from the live FleetPy vehicle, never from a future trip trace.
"""
from __future__ import annotations

from bisect import bisect_right


class FleetEventCalendar:
    def __init__(self, windows, order):
        self.windows = dict(windows)
        self.order = {int(vid): i for i, vid in enumerate(order)}
        self.starts = sorted((float(a), int(v)) for v, (a, _) in windows.items())
        self.ends = sorted((float(b), int(v)) for v, (_, b) in windows.items())
        self.start_times = [a for a, _ in self.starts]
        self._start_cursor = self._end_cursor = 0
        self._active = set()
        self._now = float("-inf")

    def active_ids(self, now):
        now = float(now)
        if now < self._now:
            raise ValueError("fleet event calendar cannot run backwards")
        while self._start_cursor < len(self.starts) and self.starts[self._start_cursor][0] <= now:
            self._active.add(self.starts[self._start_cursor][1])
            self._start_cursor += 1
        while self._end_cursor < len(self.ends) and self.ends[self._end_cursor][0] <= now:
            self._active.discard(self.ends[self._end_cursor][1])
            self._end_cursor += 1
        self._now = now
        # Preserve the original runtime insertion order (candidate tie policy).
        return tuple(sorted(self._active, key=self.order.__getitem__))

    def horizon_ids(self, now, horizon):
        active = self.active_ids(now)
        right = bisect_right(self.start_times, float(now) + float(horizon))
        forthcoming = (v for _, v in self.starts[self._start_cursor:right]
                       if self.windows[v][1] > now)
        return tuple(sorted((*active, *forthcoming)))

    def inside(self, vid, now):
        start, end = self.windows[int(vid)]
        return start <= now < end

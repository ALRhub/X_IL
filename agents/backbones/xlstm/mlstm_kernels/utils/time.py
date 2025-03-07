#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from time import time
from typing import List

""" The code in this file is adapted from https://github.com/BenediktAlkin/KappaProfiler.

Usage:
```
with Stopwatch() as sw:
    # some operation
    ...
print(f"operation took {sw.elapsed_milliseconds} milliseconds")
print(f"operation took {sw.elapsed_seconds} seconds")
```

"""

FORMAT_DATETIME_SHORT = "%y%m%d_%H%M%S"
FORMAT_DATETIME_MID = "%Y-%m-%d %H:%M:%S"


class TimeProvider:
    @staticmethod
    def time() -> float:
        return time()


class Stopwatch:
    """This class enables the user to profile specific parts of the code in a elegant way with a contextmanager.
    It supports simple start() and stop() as well as measuring lap() times, which could be useful for loops.
    """

    def __init__(self, time_provider=None):
        self._start_time: float = None
        self._lap_elapsed_seconds: list[float] = []
        self._total_elapsed_seconds: float = 0.0
        self._lap_start_time: float = None
        self._time_provider: TimeProvider = time_provider or TimeProvider()

    def start(self) -> "Stopwatch":
        assert self._start_time is None, "can't start running stopwatch"
        self._start_time = self._time_provider.time()
        self._lap_start_time = self._start_time
        return self

    def stop(self) -> "Stopwatch":
        assert self._start_time is not None, "can't stop a stopped stopwatch"
        self._total_elapsed_seconds = self._time_provider.time() - self._start_time
        self._start_time = None
        self._lap_start_time = None
        return self._total_elapsed_seconds

    def lap(self) -> float:
        assert self._start_time is not None, "lap requires stopwatch to be started"
        lap_time = self._time_provider.time() - self._lap_start_time
        self._lap_elapsed_seconds.append(lap_time)
        self._lap_start_time = self._time_provider.time()
        return lap_time

    @property
    def last_lap_time(self) -> float:
        assert len(self._lap_elapsed_seconds) > 0, "last_lap_time requires lap()/stop() to be called at least once"
        return self._lap_elapsed_seconds[-1]

    @property
    def lap_count(self) -> int:
        return len(self._lap_elapsed_seconds)

    @property
    def average_lap_time(self) -> float:
        assert len(self._lap_elapsed_seconds) > 0, "average_lap_time requires lap()/stop() to be called at least once"
        return sum(self._lap_elapsed_seconds) / len(self._lap_elapsed_seconds)

    def __enter__(self) -> "Stopwatch":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @property
    def elapsed_time(self) -> float:
        return self.elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        assert self._start_time is None, "elapsed_seconds requires stopwatch to be stopped"
        # assert len(self._lap_elapsed_seconds) > 0, "elapsed_seconds requires stopwatch to have been started and stopped"
        return self._total_elapsed_seconds

    @property
    def lap_times_seconds(self) -> list[float]:
        return self._lap_elapsed_seconds

    @property
    def elapsed_milliseconds(self) -> float:
        return self.elapsed_seconds * 1000

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600.0

    @property
    def elapsed_time_string(self) -> str:
        elapsed_seconds = self.elapsed_seconds
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60
        return f"{hours:02d}h:{minutes:02d}m:{seconds:02.3f}s"

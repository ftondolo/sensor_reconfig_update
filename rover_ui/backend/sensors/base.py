"""Shared infrastructure for sensor worker threads.

Each sensor runs in its own daemon thread and publishes its latest result into
a single-slot buffer. Consumers (HTTP/WebSocket handlers) read the latest value
without ever blocking the producer — this keeps the Jetson responsive and means
one slow/stalled sensor can never hold up another.
"""
import threading
import time
from typing import Any, Optional


class LatestValue:
    """A thread-safe single-slot buffer holding only the most recent value.

    Readers always get the freshest sample; there is no queue to back up, so a
    slow consumer cannot induce latency on the producer. A condition variable
    lets WebSocket consumers wait for the *next* frame instead of polling.
    """

    def __init__(self):
        self._value: Optional[Any] = None
        self._seq = 0
        self._cond = threading.Condition()

    def set(self, value: Any) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return self._value, self._seq

    def wait_for_next(self, last_seq: int, timeout: float = 1.0):
        """Block until a value newer than ``last_seq`` is available.

        Returns ``(value, seq)`` or ``(None, last_seq)`` on timeout.
        """
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._seq <= last_seq:
                return None, last_seq
            return self._value, self._seq


class SensorThread(threading.Thread):
    """Base class for a self-contained sensor producer.

    Subclasses implement :meth:`open`, :meth:`read_once`, and :meth:`close`.
    The run loop owns reconnection: if a sensor is missing or errors out it
    backs off and retries (or falls back to mock data) without taking down the
    rest of the system.
    """

    name: str = "sensor"

    def __init__(self, allow_mock: bool = True):
        super().__init__(daemon=True)
        self.allow_mock = allow_mock
        self.latest = LatestValue()
        self._stop = threading.Event()
        self._status = "init"  # init | opening | streaming | mock | error
        self._status_detail = ""
        self._last_error = ""

    # ---- lifecycle hooks implemented by subclasses ----------------------
    def open(self) -> None:
        raise NotImplementedError

    def read_once(self) -> Any:
        """Return one fresh sample, or None if nothing is available yet."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def read_mock(self) -> Any:
        """Optional synthetic sample so the UI works without hardware."""
        return None

    # ---- status reporting ----------------------------------------------
    def set_status(self, status: str, detail: str = "") -> None:
        self._status = status
        self._status_detail = detail

    def status(self) -> dict:
        return {
            "name": self.name,
            "status": self._status,
            "detail": self._status_detail,
            "error": self._last_error,
        }

    # ---- run loop -------------------------------------------------------
    def run(self) -> None:
        backoff = 1.0
        opened = False
        while not self._stop.is_set():
            try:
                if not opened:
                    self.set_status("opening")
                    self.open()
                    opened = True
                    backoff = 1.0
                    self.set_status("streaming")
                sample = self.read_once()
                if sample is not None:
                    self.latest.set(sample)
            except Exception as exc:  # noqa: BLE001 - isolate sensor failures
                self._last_error = f"{type(exc).__name__}: {exc}"
                try:
                    self.close()
                except Exception:
                    pass
                opened = False
                if self.allow_mock:
                    self.set_status("mock", "hardware unavailable, using synthetic data")
                    self._run_mock_until_recover(backoff)
                else:
                    self.set_status("error", self._last_error)
                    self._stop.wait(backoff)
                backoff = min(backoff * 2, 10.0)

    def _run_mock_until_recover(self, retry_after: float) -> None:
        """Emit mock samples for ``retry_after`` seconds, then let run() retry."""
        deadline = time.monotonic() + retry_after
        while not self._stop.is_set() and time.monotonic() < deadline:
            sample = self.read_mock()
            if sample is not None:
                self.latest.set(sample)
            self._stop.wait(0.05)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.close()
        except Exception:
            pass

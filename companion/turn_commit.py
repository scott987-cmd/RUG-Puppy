"""A non-blocking cancellation decision for one synchronous inference turn."""

import threading


class TurnCancelled(RuntimeError):
    """The turn was cancelled before its persistent commit was accepted."""


class TurnCommit:
    """Cancel while generating/preparing; never undo an accepted commit.

    The lock protects only a state transition, never model or SQLite work.
    ``cancel`` is safe to call directly from the event loop. The final transition
    to ``committing`` is the commit boundary: cancellation before it wins and
    rolls the transaction back; after it, the accepted commit finishes normally.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "open"

    def cancel(self) -> bool:
        """Return whether cancellation won; repeated cancellation is harmless."""
        with self._lock:
            if self._state == "open":
                self._state = "cancelled"
            return self._state == "cancelled"

    def check(self):
        with self._lock:
            if self._state == "cancelled":
                raise TurnCancelled("本轮已取消，不提交记忆")
            if self._state != "open":
                raise RuntimeError("本轮提交对象不能重复使用")

    def commit(self, operation):
        with self._lock:
            if self._state == "cancelled":
                raise TurnCancelled("本轮已取消，不提交记忆")
            if self._state != "open":
                raise RuntimeError("本轮提交对象不能重复使用")
            self._state = "committing"
        try:
            result = operation()
        except BaseException:
            with self._lock:
                self._state = "failed"
            raise
        with self._lock:
            self._state = "committed"
        return result

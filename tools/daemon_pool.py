"""Daemon-thread executor for best-effort work.

``ThreadPoolExecutor`` deliberately uses non-daemon workers and registers
them for an unconditional interpreter-exit join. That is the right contract
for ordinary application work, but it lets a wedged network call keep Hermes
alive forever after the CLI has otherwise shut down.

This executor implements the public :mod:`concurrent.futures` contract using
daemon workers. It intentionally does not subclass ``ThreadPoolExecutor`` or
copy its private worker protocol: those internals changed in Python 3.15.
Use it only for work that may safely be abandoned at process exit.
"""

from __future__ import annotations

import itertools
import os
import queue
import threading
import weakref
from concurrent.futures import Executor, Future
from typing import Any, Callable

__all__ = ["DaemonThreadPoolExecutor"]


_executor_counter = itertools.count()
_STOP = object()


class _WorkItem:
    def __init__(
        self,
        future: Future,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        if not self.future.set_running_or_notify_cancel():
            return
        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)


def _worker(
    executor_reference: weakref.ReferenceType[DaemonThreadPoolExecutor],
    work_queue: queue.SimpleQueue,
    idle_semaphore: threading.Semaphore,
    initializer: Callable[..., Any] | None,
    initargs: tuple[Any, ...],
) -> None:
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            executor = executor_reference()
            if executor is not None:
                executor._initializer_failed()
            return

    while True:
        work_item = work_queue.get()
        if work_item is _STOP:
            # Pass the sentinel to every other worker. It is harmless if no
            # workers remain and avoids needing one queue entry per thread.
            work_queue.put(_STOP)
            return
        work_item.run()
        del work_item
        idle_semaphore.release()


class DaemonThreadPoolExecutor(Executor):
    """An ``Executor`` whose reusable workers cannot block process exit."""

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        if max_workers is None:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
            max_workers = min(32, cpu_count + 4)
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")

        self._max_workers = max_workers
        self._work_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._idle_semaphore = threading.Semaphore(0)
        self._threads: set[threading.Thread] = set()
        self._shutdown = False
        self._broken: str | None = None
        self._shutdown_lock = threading.Lock()
        self._initializer = initializer
        self._initargs = initargs
        self._thread_name_prefix = thread_name_prefix or (
            f"DaemonThreadPoolExecutor-{next(_executor_counter)}"
        )

    def submit(self, fn, /, *args, **kwargs) -> Future:
        with self._shutdown_lock:
            if self._broken:
                raise RuntimeError(self._broken)
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            future = Future()
            self._work_queue.put(_WorkItem(future, fn, args, kwargs))
            self._adjust_thread_count()
            return future

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return
        if len(self._threads) >= self._max_workers:
            return

        def wake_workers(_, work_queue=self._work_queue) -> None:
            work_queue.put(_STOP)

        executor_reference = weakref.ref(self, wake_workers)
        thread = threading.Thread(
            name=f"{self._thread_name_prefix}_{len(self._threads)}",
            target=_worker,
            args=(
                executor_reference,
                self._work_queue,
                self._idle_semaphore,
                self._initializer,
                self._initargs,
            ),
            daemon=True,
        )
        thread.start()
        self._threads.add(thread)

    def _initializer_failed(self) -> None:
        with self._shutdown_lock:
            self._broken = "a thread initializer failed"
            while True:
                try:
                    work_item = self._work_queue.get_nowait()
                except queue.Empty:
                    break
                if work_item is _STOP:
                    self._work_queue.put(_STOP)
                    break
                work_item.future.set_exception(RuntimeError(self._broken))

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._shutdown_lock:
            self._shutdown = True
            if cancel_futures:
                while True:
                    try:
                        work_item = self._work_queue.get_nowait()
                    except queue.Empty:
                        break
                    if work_item is _STOP:
                        continue
                    work_item.future.cancel()
            self._work_queue.put(_STOP)

        if wait:
            for thread in self._threads:
                thread.join()

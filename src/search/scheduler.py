"""Stage 8-E multi-task batch scheduler (torch-free).

Runs many independent task generators (self-play games, evaluation games) and batches
their network requests. A task is a generator that yields ``EvalRequest(evaluator,
snapshot)``, receives the validated ``EvaluationResult`` for it and finally returns its
result. Each task has at most one outstanding request, so every task runs exactly the
steps it would run alone (``drive_requests``); only the order *between* tasks changes.
With an evaluator whose output for a snapshot does not depend on the rest of the batch,
the results are identical to running the tasks one after another.

Requests are grouped per evaluator object. An evaluator may be synchronous
(``evaluate_batch``) or asynchronous (``submit(snapshots) -> handle`` with
``handle.done()`` and ``collect(handle)``); with an asynchronous evaluator the CPU keeps
advancing other tasks while a batch is on the device.

Submission policy (no artificial waiting):
- a queue is submitted when it reaches ``max_batch``, or when no task has CPU work left;
- ``eager=True`` additionally submits whenever that evaluator is idle, so the device starts
  on small batches while the CPU works (useful only with an asynchronous evaluator).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter

from .evaluator import EvalRequest, EvaluatorOutputError, validate_evaluation

_START = object()


class _DoneHandle:
    __slots__ = ('results',)

    def __init__(self, results):
        self.results = results

    def done(self) -> bool:
        return True


def _submit(evaluator, snapshots):
    if hasattr(evaluator, 'submit'):
        return evaluator.submit(snapshots)
    return _DoneHandle(evaluator.evaluate_batch(snapshots))


def _collect(evaluator, handle):
    if isinstance(handle, _DoneHandle):
        return handle.results
    return evaluator.collect(handle)


@dataclass
class SchedulerStats:
    """Batch-size histogram and timing of one scheduler run."""

    batch_sizes: list[int] = field(default_factory=list)
    active_at_submit: list[int] = field(default_factory=list)
    device_seconds: float = 0.0     # submit -> results collected, summed over batches
    wait_seconds: float = 0.0       # time the CPU had nothing to do but wait for a batch
    wall_seconds: float = 0.0
    tasks: int = 0

    def summary(self) -> dict:
        sizes = sorted(self.batch_sizes)
        requests = sum(sizes)

        def quantile(q):
            return sizes[min(len(sizes) - 1, int(q * len(sizes)))] if sizes else None

        histogram: dict[int, int] = {}
        for size in self.batch_sizes:
            histogram[size] = histogram.get(size, 0) + 1
        return {
            'tasks': self.tasks, 'batches': len(sizes), 'requests': requests,
            'mean_batch': requests / len(sizes) if sizes else None,
            # request-weighted mean: the batch size an average request was evaluated in
            'request_weighted_batch': (sum(s * s for s in sizes) / requests) if requests else None,
            'p50_batch': quantile(0.5), 'p95_batch': quantile(0.95),
            'max_batch': sizes[-1] if sizes else None,
            'histogram': {str(k): v for k, v in sorted(histogram.items())},
            'mean_active_at_submit': (sum(self.active_at_submit) / len(self.active_at_submit)
                                      if self.active_at_submit else None),
            'device_seconds': self.device_seconds, 'wait_seconds': self.wait_seconds,
            'wall_seconds': self.wall_seconds,
        }


def run_tasks(tasks: Iterable[Callable[[], object]], *, max_active: int | None = None,
              max_batch: int | None = None, eager: bool = False,
              stats: SchedulerStats | None = None) -> list:
    """Run task factories (each returns a fresh task generator); results in task order.

    ``max_active`` bounds the tasks alive at once; finished tasks are replaced by the next
    factory (refill), so long runs keep the batch full. ``None`` starts every task at once.
    """
    if max_active is not None and max_active < 1:
        raise ValueError('max_active must be positive')
    if max_batch is not None and max_batch < 1:
        raise ValueError('max_batch must be positive')
    stats = stats if stats is not None else SchedulerStats()
    started = perf_counter()
    factories = iter(enumerate(tasks))
    results: dict[int, object] = {}
    active: dict[int, object] = {}
    ready: deque = deque()               # (task id, value to send)
    queues: dict[int, list] = {}         # evaluator id -> [(task id, snapshot)]
    evaluators: dict[int, object] = {}
    inflight: dict[int, tuple] = {}      # evaluator id -> (handle, task ids, snapshots, t0)
    exhausted = False

    def refill():
        nonlocal exhausted
        while not exhausted and (max_active is None or len(active) < max_active):
            try:
                task_id, factory = next(factories)
            except StopIteration:
                exhausted = True
                return
            active[task_id] = factory()
            ready.append((task_id, _START))
            stats.tasks += 1

    def submit(key):
        queue = queues[key]
        take = len(queue) if max_batch is None else min(len(queue), max_batch)
        batch, queues[key] = queue[:take], queue[take:]
        ids = [task_id for task_id, _ in batch]
        snapshots = [snapshot for _, snapshot in batch]
        stats.batch_sizes.append(len(batch))
        stats.active_at_submit.append(len(active))
        inflight[key] = (_submit(evaluators[key], snapshots), ids, snapshots, perf_counter())

    def collect(key):
        handle, ids, snapshots, t0 = inflight.pop(key)
        batch_results = _collect(evaluators[key], handle)
        stats.device_seconds += perf_counter() - t0
        if not isinstance(batch_results, list) or len(batch_results) != len(snapshots):
            raise EvaluatorOutputError('evaluate_batch must return one result per snapshot')
        for task_id, snapshot, result in zip(ids, snapshots, batch_results):
            ready.append((task_id, validate_evaluation(result, snapshot)))

    refill()
    while active:
        for key in [k for k, (handle, *_rest) in inflight.items() if handle.done()]:
            collect(key)
        if ready:
            task_id, value = ready.popleft()
            task = active[task_id]
            try:
                request = next(task) if value is _START else task.send(value)
            except StopIteration as stop:
                results[task_id] = stop.value
                del active[task_id]
                refill()
                continue
            if not isinstance(request, EvalRequest):
                raise TypeError('tasks must yield EvalRequest')
            key = id(request.evaluator)
            evaluators[key] = request.evaluator
            queues.setdefault(key, []).append((task_id, request.snapshot))
            if key not in inflight and (
                    eager or (max_batch is not None and len(queues[key]) >= max_batch)):
                submit(key)
            continue
        idle = [k for k, q in queues.items() if q and k not in inflight]
        for key in idle:
            submit(key)
        if not idle and inflight:
            # Nothing for the CPU to do: block on the oldest batch.
            key = min(inflight, key=lambda k: inflight[k][3])
            waited = perf_counter()
            collect(key)
            stats.wait_seconds += perf_counter() - waited
        elif not idle and not inflight:
            raise RuntimeError('scheduler stalled: active tasks without requests')
    stats.wall_seconds += perf_counter() - started
    return [results[i] for i in range(len(results))]

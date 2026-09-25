"""Rollout infrastructure primitives borrowed from DeepSeek Elastic Compute (DSec, arXiv:2609.22978).

* Snapshot / branch (DSec `pack_diff` checkpoint-restore, pause/resume): `fork_branches` forks the current
  process at a decision point. Each child inherits the live game (FarmEnv) AND every heuristic expert's
  internal Python state copy-on-write (the analogue of sharing one host page-cache copy across guests), applies
  its own choice and continues; results come back over a pipe. Branch cost is O(remaining steps) instead of
  O(replay the whole prefix), and nothing has to be serialisable.
* Warm pool (DSec GPU FnCall "warm pool of Python processes initialises the runtime and imports libraries in
  advance"): `warm_pool` preloads heavy modules / compiled agent code in the parent and forks workers, so
  tasks start without spawn + import cost and share read-only memory.
* QoS classes (DSec SCHED_IDLE + core scheduling): `best_effort()` puts bulk rollouts under SCHED_IDLE so that
  latency-sensitive work (per-step latency benchmarks) keeps its budget; `pin(cores)` isolates a
  latency-sensitive process on dedicated cores.
"""
import multiprocessing as mp
import os
import pickle
import select
import struct


def best_effort():
    """Lower this process to SCHED_IDLE (yields the CPU whenever a normal task is runnable)."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, PermissionError, OSError):
        os.nice(19)


def pin(cores):
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError):
        pass


def _send(fd, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    os.write(fd, struct.pack("<Q", len(data)))
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _recv(fd):
    head = b""
    while len(head) < 8:
        chunk = os.read(fd, 8 - len(head))
        if not chunk:
            raise EOFError
        head += chunk
    n = struct.unpack("<Q", head)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, min(1 << 20, n - len(buf)))
        if not chunk:
            raise EOFError
        buf += chunk
    return pickle.loads(bytes(buf))


def fork_branches(choices, run_branch, max_parallel=None):
    """For each choice c: fork, call run_branch(c) in the child (it may mutate any live state freely), return
    [result per choice] in order. Children run concurrently (at most max_parallel at a time)."""
    max_parallel = max_parallel or os.cpu_count() or 1
    results = [None] * len(choices)
    pending = list(enumerate(choices))
    live = {}                                                  # read fd -> (index, pid)
    while pending or live:
        while pending and len(live) < max_parallel:
            i, c = pending.pop(0)
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:                                       # child: branch from the shared snapshot
                os.close(r)
                try:
                    out = ("ok", run_branch(c))
                except BaseException as e:                     # report, never hang the parent
                    out = ("err", repr(e))
                try:
                    _send(w, out)
                finally:
                    os._exit(0)
            os.close(w)
            live[r] = (i, pid)
        ready, _, _ = select.select(list(live), [], [])
        for r in ready:
            i, pid = live.pop(r)
            try:
                status, val = _recv(r)
            except EOFError:
                status, val = "err", "child died"
            os.close(r)
            os.waitpid(pid, 0)
            results[i] = val if status == "ok" else RuntimeError(val)
    return results


def warm_pool(n, initializer=None, initargs=()):
    """Pool of forked workers that inherit everything already imported/compiled in the parent."""
    return mp.get_context("fork").Pool(n, initializer=initializer, initargs=initargs)

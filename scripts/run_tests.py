#!/usr/bin/env python3
"""Run the camera-free test suite in parallel worker processes.

``python -m unittest discover`` runs every test in one process, and about half
of its wall time is spent waiting (worker threads, timer ticks, Tk and Qt event
loops) rather than computing. This runner discovers the tests, groups them by
test class, spreads the groups over several ``python -m unittest`` processes
and, unless told otherwise, measures line coverage in every worker and merges
the results, so the suite and the 100% gate behave as in the serial run.

Usage (from anywhere; only the standard library is needed, plus ``coverage``
for the coverage gate):

    python scripts/run_tests.py                  # parallel, with the 100% coverage gate
    python scripts/run_tests.py --no-coverage    # fastest: tests only
    python scripts/run_tests.py -j 4             # choose the number of workers
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
SRC_DIR = ROOT / "src"

def default_jobs() -> int:
    """Pick a worker count that scales with the machine running the tests.

    Uses the CPUs this process may actually run on (honouring CPU affinity and
    container limits where Python can see them), not just the installed total.
    There is no fixed cap: ``partition`` never makes more shards than there are
    test classes, so a large machine simply gets one class per worker.

    Args:
        None.

    Returns:
        The number of usable CPUs, at least 1.
    """
    counter = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if counter is not None:
        return max(1, counter() or 1)
    affinity = getattr(os, "sched_getaffinity", None)  # Linux
    if affinity is not None:
        return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


def flatten(suite: unittest.TestSuite):
    """Yield every individual test in a (possibly nested) suite.

    Args:
        suite: A suite returned by ``TestLoader.discover``.

    Returns:
        A generator of ``unittest.TestCase`` instances.
    """
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def discover_groups() -> dict[str, int]:
    """Discover the tests and count them per test class.

    Args:
        None.

    Returns:
        A mapping of ``module.ClassName`` to the number of tests in that class.

    Raises:
        SystemExit: A test module failed to import, so the serial runner should
            be used to see the error.
    """
    sys.path[:0] = [str(SRC_DIR), str(TESTS_DIR)]
    suite = unittest.defaultTestLoader.discover(str(TESTS_DIR), top_level_dir=str(TESTS_DIR))
    groups: dict[str, int] = {}
    for test in flatten(suite):
        cls = type(test)
        if cls.__name__ == "_FailedTest":
            raise SystemExit(
                f"Could not import {test.id()}; run `python -m unittest discover -s tests` to see why."
            )
        key = f"{cls.__module__}.{cls.__qualname__}"
        groups[key] = groups.get(key, 0) + 1
    return groups


def partition(groups: dict[str, int], jobs: int) -> list[list[str]]:
    """Split test classes into balanced shards, largest first.

    Greedy longest-processing-time assignment using the test count as the cost
    estimate: each class goes to the shard with the smallest total so far.

    Args:
        groups: Test class to test count, as returned by ``discover_groups``.
        jobs: How many shards to make.

    Returns:
        Non-empty shards, each a list of ``module.ClassName`` strings.
    """
    shards: list[list[str]] = [[] for _ in range(max(1, jobs))]
    loads = [0] * len(shards)
    for name, count in sorted(groups.items(), key=lambda item: (-item[1], item[0])):
        index = loads.index(min(loads))
        shards[index].append(name)
        loads[index] += count
    return [shard for shard in shards if shard]


def worker_env(coverage_file: Path | None) -> dict[str, str]:
    """Build the environment every worker process runs with.

    Args:
        coverage_file: Base path for coverage data, or ``None`` when coverage is off.

    Returns:
        A copy of ``os.environ`` with ``PYTHONPATH`` pointing at ``src`` and
        ``tests`` and Qt set to its offscreen platform unless already chosen.
    """
    env = dict(os.environ)
    paths = [str(SRC_DIR), str(TESTS_DIR)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    if coverage_file is not None:
        env["COVERAGE_FILE"] = str(coverage_file)
    return env


def run_shard(classes: list[str], env: dict[str, str], coverage: bool) -> tuple[int, str]:
    """Run one shard of test classes in a fresh interpreter.

    Args:
        classes: ``module.ClassName`` strings for ``python -m unittest``.
        env: Environment from ``worker_env``.
        coverage: Whether to measure coverage (parallel mode, one data file per process).

    Returns:
        ``(return code, combined stdout and stderr)``.
    """
    command = [sys.executable]
    if coverage:
        command += ["-m", "coverage", "run", "--parallel-mode"]
    command += ["-m", "unittest", *classes]
    done = subprocess.run(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return done.returncode, done.stdout


def coverage_gate(env: dict[str, str]) -> int:
    """Merge the workers' coverage data and apply the ``.coveragerc`` gate.

    Args:
        env: Environment from ``worker_env``, which names the coverage data file.

    Returns:
        0 when line coverage meets ``fail_under``, otherwise the failing return code.
    """
    base = [sys.executable, "-m", "coverage"]
    combined = subprocess.run(base + ["combine", "--quiet"], cwd=ROOT, env=env)
    if combined.returncode != 0:
        return combined.returncode
    return subprocess.run(base + ["report"], cwd=ROOT, env=env).returncode


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments without the program name, or ``None`` for ``sys.argv``.

    Returns:
        The parsed namespace with ``jobs`` and ``no_coverage``.
    """
    parser = argparse.ArgumentParser(description="Run the test suite in parallel worker processes.")
    parser.add_argument(
        "-j", "--jobs", type=int, default=default_jobs(), help="worker processes (default: usable CPU cores)"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="skip coverage measurement and the 100%% gate (faster)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Discover, shard and run the tests, then apply the coverage gate.

    Args:
        argv: Arguments without the program name, or ``None`` for ``sys.argv``.

    Returns:
        0 when every test passes (and coverage is 100% unless disabled), otherwise 1.
    """
    args = parse_args(argv)
    coverage = not args.no_coverage
    if coverage and importlib.util.find_spec("coverage") is None:
        print("coverage is not installed: `pip install coverage`, or pass --no-coverage.", file=sys.stderr)
        return 1

    started = time.perf_counter()
    groups = discover_groups()
    shards = partition(groups, args.jobs)
    total = sum(groups.values())

    with tempfile.TemporaryDirectory(prefix="espi-tests-") as scratch:
        env = worker_env(Path(scratch) / ".coverage" if coverage else None)
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            results = list(pool.map(lambda shard: run_shard(shard, env, coverage), shards))

        failed = [(shard, output) for shard, (code, output) in zip(shards, results) if code != 0]
        for shard, output in failed:
            print(f"--- FAILED shard ({len(shard)} classes) ---\n{output}")
        status = 1 if failed else 0
        if coverage and not failed:
            status = coverage_gate(env)

    elapsed = time.perf_counter() - started
    outcome = "OK" if status == 0 else "FAILED"
    print(f"{outcome}: {total} tests in {len(groups)} classes on {len(shards)} workers, {elapsed:.1f}s")
    return status


if __name__ == "__main__":
    sys.exit(main())

"""Load a Kaggle agent file exactly the way kaggle_environments does.

`kaggle_environments.agent.get_last_callable` execs the file and returns the LAST callable in the namespace
dict. Dict order is the order in which names were *first* bound, so the entry point is not necessarily the
global `agent` (e.g. cha22 -> `ig_agent`, v48 -> `_e335_agent`). Evaluating `module.agent` instead silently
plays an older layer of such agents.
"""
import importlib.util
import inspect
import os
import sys
import uuid


def load_module(path):
    """Exec an agent file as a fresh module. Like kaggle_environments, the file's directory is on sys.path
    while it executes (multi-file submissions import sibling modules / compiled extensions); the working
    directory is switched there too, for agents that open data files by relative path."""
    name = "kagent_" + uuid.uuid4().hex
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    cwd = os.getcwd()
    sys.path.insert(0, d)
    try:
        os.chdir(d)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cwd)
        if sys.path and sys.path[0] == d:
            sys.path.pop(0)
        if d not in sys.path:                  # agents may import siblings lazily at call time
            sys.path.append(d)
    return m


def entry_name(module):
    """Name of the callable Kaggle would pick (last callable in first-binding order)."""
    names = [k for k, v in vars(module).items() if callable(v) and not k.startswith("__")]
    return names[-1]


def call_adapter(fn):
    """obs, cfg -> action, whatever the arity of `fn`."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 2
    return fn if n >= 2 else (lambda obs, cfg=None, f=fn: f(obs))


def load_agent(path):
    """-> callable(obs, cfg) that is exactly the agent Kaggle would run."""
    m = load_module(path)
    return call_adapter(getattr(m, entry_name(m)))

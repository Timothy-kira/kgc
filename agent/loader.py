"""Load a Kaggle agent file exactly the way kaggle_environments does.

`kaggle_environments.agent.get_last_callable` execs the file and returns the LAST callable in the namespace
dict. Dict order is the order in which names were *first* bound, so the entry point is not necessarily the
global `agent` (e.g. cha22 -> `ig_agent`, v48 -> `_e335_agent`). Evaluating `module.agent` instead silently
plays an older layer of such agents.
"""
import importlib.util
import inspect
import sys
import uuid


def load_module(path):
    name = "kagent_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
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

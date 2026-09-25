"""Load this agent's local settings and construct its farming policy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

def _source_directory():
    filename = globals().get("__file__") or _source_directory.__code__.co_filename
    return Path(filename).resolve().parent

_root = _source_directory()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
with (_root / "settings.json").open(encoding="utf-8") as _stream:
    _settings = json.load(_stream)
_name = "_notebook_policy_" + hashlib.sha256(str(_root).encode()).hexdigest()[:16]
_spec = importlib.util.spec_from_file_location(_name, _root / "policy.py")
_policy = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _policy
_spec.loader.exec_module(_policy)
_candidate = getattr(_policy, "build_agent")(_settings)
if not callable(_candidate):
    raise TypeError("The policy factory must return a callable agent")

def agent(observation, configuration=None):
    return _candidate(observation, configuration)

agent.telemetry = getattr(_candidate, "telemetry", {})

"""Frozen V36 baseline with optional pre-investment scenario admission."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import runpy

from sheep_admission import evaluate_admission

_ROOT = Path(globals().get("__file__", inspect.currentframe().f_code.co_filename)).resolve().parent
_CONFIG = json.loads((_ROOT / "candidate_config.json").read_text())
_SOURCE = _ROOT / "upstream.py"
if hashlib.sha256(_SOURCE.read_bytes()).hexdigest() != _CONFIG["upstream_sha256"]:
    raise RuntimeError("Frozen upstream source identity mismatch")
_LOADED = runpy.run_path(str(_SOURCE))
_PARENT = _LOADED["agent"]
_UPSTREAM = _PARENT.__globals__
_ORIGINAL_ELIGIBLE = _UPSTREAM["_v233_eligible"]
_REPORT = {"candidate": _CONFIG["candidate"], "parent_eligible": 0,
           "admission_allow": 0, "admission_decline": 0,
           "admission_fallback": 0, "admission_errors": 0,
           "normalized_empty_slots": 0, "last_admission": None,
           "upstream_error_counters": {}}


def _admission_gate(observation, native):
    eligible = _ORIGINAL_ELIGIBLE(observation, native)
    if not eligible:
        return False
    _REPORT["parent_eligible"] += 1
    if _CONFIG["candidate"] == "A":
        return True
    try:
        hires = {}
        for day in range(12, 30):
            # The frozen source unconditionally enters route2 at step648.
            # This is its known own schedule, not a predicted future shop.
            planned_native = dict(native, route=2) if day >= 27 else native
            plan = _UPSTREAM["_v219_native_day"](planned_native, day)
            hires[str(day)] = sum(bool(order) and order[0] == "HIRE"
                                  for action in plan for order in action.get("market", []))
        params = copy.deepcopy(_UPSTREAM["_R37_MARKET_PARAMS"])
        for item, patch in observation["market"].get("params", {}).items():
            if item in params:
                params[item].update(patch)
        result = evaluate_admission(observation, hires, params=params,
                                    scenario_config=_CONFIG["scenario_model"])
        _REPORT["last_admission"] = result
        if result["decision"] == "DECLINE":
            _REPORT["admission_decline"] += 1
            return False
        if result["decision"] == "ALLOW":
            _REPORT["admission_allow"] += 1
        else:
            _REPORT["admission_fallback"] += 1
        return True
    except Exception as error:
        _REPORT["admission_errors"] += 1
        _REPORT["last_admission"] = {"status": "ERROR", "decision": "PARENT_FALLBACK",
                                      "reason": type(error).__name__ + ": " + str(error)}
        return True


def _upstream_errors():
    result = {}
    for name, value in _UPSTREAM.items():
        if name.startswith("_") and isinstance(value, dict):
            for key, count in value.items():
                if isinstance(key, str) and any(token in key.lower() for token in ("error", "fallback")) and isinstance(count, (int, float)):
                    result[name + "." + key] = count
    chassis = getattr(_UPSTREAM.get("_IMPL"), "chassis", None)
    diagnostics = getattr(chassis, "diagnostics", {})
    _REPORT["upstream_chassis_diagnostics"] = copy.deepcopy(diagnostics)
    for key, count in diagnostics.items():
        if isinstance(key, str) and any(token in key.lower() for token in ("error", "fallback")) and isinstance(count, (int, float)):
            result["chassis." + key] = count
    return result


def _normalize_action(action):
    result = copy.deepcopy(action)
    for index, order in enumerate(result.get("market", [])):
        if order == []:
            result["market"][index] = ["SELL", "WHEAT", 0]
            _REPORT["normalized_empty_slots"] += 1
    return result


_UPSTREAM["_v233_eligible"] = _admission_gate


def agent(observation, configuration=None):
    if int(observation.get("step", -1)) == 0:
        for key in ("parent_eligible", "admission_allow", "admission_decline",
                    "admission_fallback", "admission_errors", "normalized_empty_slots"):
            _REPORT[key] = 0
        _REPORT["last_admission"] = None
    # The upstream configuration guards define the parent's supported contract.
    # Additional admission assumptions are only valid for the pinned defaults.
    supported = configuration is None or all(configuration.get(key, default) == default
        for key, default in (("boardSize", 10), ("turnsPerDay", 24),
            ("shedCapacity", 100), ("maxMarketOrdersPerTurn", 10),
            ("farmHandCostMult", 1), ("townShopSellInterval", 4),
            ("townCenterSellInterval", 24), ("episodeSteps", 720)))
    _UPSTREAM["_v233_eligible"] = _admission_gate if supported else _ORIGINAL_ELIGIBLE
    result = _normalize_action(_PARENT(observation, configuration))
    _REPORT["upstream_error_counters"] = _upstream_errors()
    _REPORT["upstream_telemetry"] = copy.deepcopy(getattr(_PARENT, "telemetry", {}))
    _REPORT["admission_supported_configuration"] = supported
    return result


agent.telemetry = _REPORT

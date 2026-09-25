# SPDX-License-Identifier: Apache-2.0
"""Construct independent policy state for each replay or notebook example."""
import importlib.util
import itertools
from pathlib import Path
import sys

_instances = itertools.count()

def build_agent(settings):
    if set(settings) != {'sale_horizon'} or type(settings['sale_horizon']) is not int or settings['sale_horizon'] not in (1, 2, 3):
        raise ValueError('sale_horizon must be 1, 2 or 3')
    path = Path(build_agent.__code__.co_filename).resolve().parent / 'router.py'
    name = __name__ + '_router_' + str(next(_instances))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.SALE_HORIZON = settings['sale_horizon']
    return module.agent

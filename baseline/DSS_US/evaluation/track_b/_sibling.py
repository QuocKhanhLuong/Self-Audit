"""Load sibling evaluator modules whether imported as a package or by file path."""
import importlib.util
from pathlib import Path
import sys


def load(name):
    module_name = f"dss_us_track_b_{name}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module

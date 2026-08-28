"""Behavioral checks for the optional official Viser demo surface."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _load_keyboard_registration():
    """Load the helper with only the demo's imported contract stubbed."""
    package_names = (
        "scripts",
        "scripts.interactive_demo",
        "scripts.interactive_demo.gui",
    )
    common_name = "scripts.interactive_demo.common"
    instructions_name = "scripts.interactive_demo.gui.instructions"
    module_name = "scripts.interactive_demo.gui.io_for_test"

    package_modules = {}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        package_modules[name] = package
    common = types.ModuleType(common_name)
    common.viser = types.SimpleNamespace(ClientHandle=object)
    common.__all__ = ["viser"]
    instructions = types.ModuleType(instructions_name)
    instructions.KEYBOARD_SHORTCUTS_MD = ""
    previous = {name: sys.modules.get(name) for name in (*package_names, common_name, instructions_name)}
    sys.modules.update({**package_modules, common_name: common, instructions_name: instructions})
    source = Path(__file__).parents[1] / "scripts/interactive_demo/gui/io.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "scripts.interactive_demo.gui"
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module._register_keyboard_handlers
    finally:
        sys.modules.pop(module_name, None)
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


class _SceneWithoutKeyboard:
    pass


class _SceneWithKeyboard:
    def __init__(self) -> None:
        self.registered: list[tuple[str, int, Any]] = []

    def on_keyboard_event(self, event_type: str, *, debounce_ms: int):
        def register(handler: Any) -> Any:
            self.registered.append((event_type, debounce_ms, handler))
            return handler

        return register


def test_official_viser_without_keyboard_extension_is_explicitly_optional(capsys) -> None:
    register = _load_keyboard_registration()
    client = types.SimpleNamespace(scene=_SceneWithoutKeyboard())

    assert register(client, lambda _event: None) is False
    assert capsys.readouterr().out == ""


def test_keyboard_extension_still_registers_both_lifecycle_events() -> None:
    register = _load_keyboard_registration()
    scene = _SceneWithKeyboard()
    client = types.SimpleNamespace(scene=scene)
    def handler(_event: Any) -> None:
        return None

    assert register(client, handler) is True
    assert [(event_type, debounce_ms) for event_type, debounce_ms, _ in scene.registered] == [
        ("keydown", 100),
        ("keyup", 100),
    ]
    assert all(callback is handler for _, _, callback in scene.registered)

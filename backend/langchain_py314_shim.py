"""
Python 3.14 + Pydantic v2: LangChain's ``Chain`` class in ``langchain.chains.base`` annotates
fields with ``dict[str, Any]``. After the class body runs, ``dict`` in that namespace refers to
the ``dict()`` method (Pydantic model API), so evaluating ``Optional[dict[str, Any]]`` raises
``TypeError: 'function' object is not subscriptable``.

This module rewrites those annotations to ``typing.Dict`` and executes the module once, before
``langchain.agents`` (or anything else) imports ``langchain.chains.base``.

Call :func:`install` from ``main.py`` before importing ``agents`` or ``langchain.agents``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types


def install() -> None:
    if sys.version_info < (3, 14):
        return
    name = "langchain.chains.base"
    if name in sys.modules:
        return
    spec = importlib.util.find_spec(name)
    if spec is None or not getattr(spec, "origin", None):
        return
    path = spec.origin
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "dict[str," not in src:
        return
    src = src.replace(
        "from typing import Any, Optional, Union, cast",
        "from typing import Any, Dict, Optional, Union, cast",
        1,
    )
    src = src.replace("dict[str,", "Dict[str,")
    importlib.import_module("langchain.chains")
    mod = types.ModuleType(name)
    mod.__file__ = path
    mod.__package__ = "langchain.chains"
    mod.__name__ = name
    mod.__spec__ = spec
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)  # noqa: S102
    setattr(sys.modules["langchain.chains"], "base", mod)

#!/usr/bin/env python
"""
Enhanced integrity checker for the gc_analysis code-base.

Outputs JSON with:
  • missing_imports     – files & lines that import modules now missing
  • duplicate_symbols   – public names appearing in >1 module
  • file_tree           – list of every tracked .py file (relative paths)
  • symbols_by_file     – per-file map of
        { "funcs": [f1, f2], "classes": {C: [methods…]} }
"""

from __future__ import annotations
import ast, importlib.util, json, sys, traceback
from pathlib import Path

PKG = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/underlying functions"  # adjust if you renamed the package
repo = Path(__file__).resolve().parent
pkg_root = repo / PKG

missing: list[dict] = []
duplicates: dict[str, list[str]] = {}
public_map: dict[str, list[str]] = {}
file_tree: list[str] = []
symbols_by_file: dict[str, dict] = {}


# ------------------------------------------------------------------ helpers
def is_std_or_site(name: str) -> bool:
    root = name.split(".")[0]
    try:
        spec = importlib.util.find_spec(root)
    except ImportError:
        return False
    if spec is None:
        return False
    return (
        spec.origin is None
        or "site-packages" in (spec.origin or "")
        or spec.origin.startswith(sys.base_prefix)
    )


def add_duplicate(name: str, module: str) -> None:
    public_map.setdefault(name, []).append(module)


# ------------------------------------------------------------------ scan
for py in repo.rglob("*.py"):
    # skip venv / build dirs
    if any(p in py.parts for p in (".venv", "site-packages", "__pycache__")):
        continue

    rel = py.relative_to(repo)
    file_tree.append(str(rel))

    src = py.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(rel))

    # -------------------- import checks
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if node.module else []
            )
            for mod in names:
                # Only check stdlib / 3rd-party if root matches our pkg or unknown
                if mod and (mod.startswith(PKG) or not is_std_or_site(mod)):
                    try:
                        importlib.import_module(mod)
                    except Exception as exc:
                        missing.append(
                            {
                                "file": str(rel),
                                "line": node.lineno,
                                "import": mod,
                                "error": repr(exc),
                            }
                        )

    # -------------------- symbol inventory
    funcs: list[str] = []
    classes: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
            add_duplicate(node.name, str(rel))
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes[node.name] = methods
            add_duplicate(node.name, str(rel))
            for m in methods:
                add_duplicate(f"{node.name}.{m}", f"{rel}")

    symbols_by_file[str(rel)] = {"funcs": funcs, "classes": classes}

# -------------------- duplicates map
duplicate_symbols = {k: v for k, v in public_map.items() if len(v) > 1}

# -------------------- JSON out
report = {
    "missing_imports": missing,
    "duplicate_symbols": duplicate_symbols,
    "file_tree": sorted(file_tree),
    "symbols_by_file": symbols_by_file,
}
print(json.dumps(report, indent=2))

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable, List


def autodiscover(
    package: str,
    *,
    exclude: Iterable[str] = (),
    recursive: bool = False,
    include_packages: bool = False,
) -> List[str]:
    """
    Import all modules under `package` to trigger decorator-based registrations.

    Args:
        package: Dotted module path (e.g., "src.data.pipelines")
        exclude: Module base-names to skip (e.g., {"__init__", "base", "registry"})
        recursive: If True, walk subpackages as well

    Returns:
        List of imported module names.
    """
    pkg = importlib.import_module(package)
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return []

    exclude_set = set(exclude)
    prefix = pkg.__name__ + "."

    walker = (
        pkgutil.walk_packages(pkg_path, prefix=prefix)
        if recursive
        else pkgutil.iter_modules(pkg_path, prefix=prefix)
    )

    imported: List[str] = []
    for modinfo in walker:
        name = modinfo.name
        if getattr(modinfo, "ispkg", False) and not include_packages:
            continue
        base = name.rsplit(".", 1)[-1]
        if base in exclude_set:
            continue
        importlib.import_module(name)
        imported.append(name)
    return imported



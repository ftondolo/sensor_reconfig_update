# Deliberately NOT "from .navigator import Navigator" at module scope: that
# would run navigator.py's top-level code (including "from . import
# tag_fusion") WHILE this package's own __init__ is still mid-execution, i.e.
# before "backend.nav" is a fully initialized module. Some Python versions
# tolerate that reentrant partial-package state; others raise "cannot import
# name 'tag_fusion' from partially initialized module" (observed on the
# Jetson's Python 3.8, though not on newer interpreters). Deferring the import
# to first access (PEP 562) means "backend.nav" always finishes initializing
# -- trivially, since there is nothing left to do -- before navigator.py ever
# runs, so its own sibling import of tag_fusion sees a complete package.
# `from .nav import Navigator` (and `backend.nav.Navigator`) keep working
# exactly as before; only WHEN navigator.py loads has changed.
def __getattr__(name):
    if name == "Navigator":
        from .navigator import Navigator
        return Navigator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Navigator"]

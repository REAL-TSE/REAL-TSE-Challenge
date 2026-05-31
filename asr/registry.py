"""Registry of ASR backends.

Backends register themselves via the :func:`register_asr` decorator and are
instantiated by name through :func:`get_asr_model`. The registry is the only
component that ``asr/asr_inference.py`` needs to know about, so adding a new
backend never requires touching the main pipeline.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Type

from .base import ASRModel

_REGISTRY: Dict[str, Type[ASRModel]] = {}
_ALIASES: Dict[str, str] = {}


def register_asr(
    name: str,
    aliases: Optional[Iterable[str]] = None,
) -> Callable[[Type[ASRModel]], Type[ASRModel]]:
    """Class decorator to register an ASR backend."""

    def _decorator(cls: Type[ASRModel]) -> Type[ASRModel]:
        if not issubclass(cls, ASRModel):
            raise TypeError(
                f"@register_asr can only be applied to ASRModel subclasses, got {cls!r}"
            )
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"ASR backend name conflict: {name!r} already registered to "
                f"{_REGISTRY[name].__name__}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        for alias in aliases or ():
            if alias in _ALIASES and _ALIASES[alias] != name:
                raise ValueError(
                    f"ASR backend alias conflict: {alias!r} already points to "
                    f"{_ALIASES[alias]!r}, cannot rebind to {name!r}"
                )
            _ALIASES[alias] = name
        return cls

    return _decorator


def resolve_name(name: str) -> str:
    """Resolve aliases to the canonical name."""
    return _ALIASES.get(name, name)


def get_asr_model(name: str, **kwargs) -> ASRModel:
    """Instantiate an ASR backend by name.

    Extra keyword arguments are forwarded to the backend constructor
    (commonly ``device`` and ``model_path``).
    """
    canonical = resolve_name(name)
    if canonical not in _REGISTRY:
        available = ", ".join(sorted(list_models())) or "<empty>"
        raise ValueError(
            f"Unknown ASR model: {name!r}. Registered: {available}"
        )
    return _REGISTRY[canonical](**kwargs)


def list_models() -> List[str]:
    """Return all canonical model names currently registered."""
    return list(_REGISTRY.keys())

"""Backend implementations for the ASR registry.

Importing this package triggers ``@register_asr`` on every backend module.
Heavy dependencies (sherpa_onnx) are loaded lazily inside each backend's
``__init__``, so simply importing the package does not pay the cost of
every model.
"""

from __future__ import annotations

# Side-effect imports register each backend with the central registry.
from . import zipformer_sherpa  # noqa: F401

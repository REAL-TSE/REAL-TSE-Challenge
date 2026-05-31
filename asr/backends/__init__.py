"""Backend implementations for the ASR registry.

Importing this package triggers ``@register_asr`` on every backend module.
Heavy dependencies (sherpa_onnx) are loaded lazily inside each backend's
``__init__``, so simply importing the package does not pay the cost of
every model.
"""

from __future__ import annotations

# Side-effect imports register each backend with the central registry.
# Order: register the default Zipformer backends first, then the historical
# Whisper / FireRedASR backends. All heavy deps (transformers, fireredasr,
# sherpa_onnx) are imported lazily inside each backend's ``__init__``, so
# importing this package never pays the cost of an unused backend.
from . import zipformer_sherpa  # noqa: F401
from . import whisper_hf  # noqa: F401
from . import firered_aed  # noqa: F401

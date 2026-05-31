"""Abstract base class for ASR backends.

All ASR backends in REAL-TSE-Challenge must subclass :class:`ASRModel` and
implement :meth:`transcribe`. The contract is intentionally minimal so that
new backends can be added with one file under ``asr/backends/`` plus a
``@register_asr`` decorator.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


class ASRModel(abc.ABC):
    """Unified ASR backend contract.

    Parameters
    ----------
    device : str
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``. Backends that
        cannot honor the requested device should still accept the argument
        and fall back gracefully.
    model_path : Optional[str]
        Local checkpoint directory. ``None`` means the backend should resolve
        a sensible default (typically under ``./zipformer/pretrained_models``).
    """

    #: Canonical short name used by the registry (set by ``@register_asr``).
    name: str = ""

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.extra_kwargs = kwargs

    @abc.abstractmethod
    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe one audio file.

        Parameters
        ----------
        audio_path : str
            Absolute path to a readable wav file.
        language : str
            Either ``"zh"`` or ``"en"``. Backends that do not need a
            language hint may ignore it.

        Returns
        -------
        str
            Plain-text transcript. Empty string is allowed but must not be
            ``None``.
        """

    def transcribe_array(
        self,
        audio: "np.ndarray",
        sr: int,
        language: str,
    ) -> str:
        """Optional in-memory entry point. Default implementation raises."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support in-memory inference; "
            "use transcribe(audio_path, language) instead."
        )

    # Backwards-compat alias used by the historical asr_inference.py.
    def transcribe_audio(self, audio_path: str, language: str = "en") -> str:
        return self.transcribe(audio_path, language)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r} device={self.device!r}>"

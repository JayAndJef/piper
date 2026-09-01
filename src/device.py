from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


def piper_device() -> str:
    """Return the Piper execution device (``"cpu"`` or ``"cuda"``)."""
    return os.environ.get("PIPER_DEVICE", "cuda")


def is_cuda() -> bool:
    return piper_device().startswith("cuda")


def create_stream_context(device: str) -> AbstractContextManager:
    """Return a reusable context manager: ``torch.cuda.stream(...)`` for CUDA,
    ``nullcontext()`` for CPU."""
    if device == "cpu":
        return nullcontext()
    return torch.cuda.stream(torch.cuda.Stream(device=device))


def create_event() -> Any:
    if not is_cuda():
        return None
    return torch.cuda.Event()


def record_event(stream_ctx: AbstractContextManager) -> Any:
    """Create a new event and record it on *stream_ctx*. Returns the event."""
    evt = create_event()
    if is_cuda():
        evt.record(stream_ctx.stream)
    return evt


def wait_event(stream_ctx: AbstractContextManager, evt: Any) -> None:
    """Make *stream_ctx* wait for *evt*. No-op on CPU or when *evt* is ``None``."""
    if evt is not None and is_cuda():
        stream_ctx.stream.wait_event(evt)


def device_synchronize() -> None:
    if is_cuda():
        torch.cuda.synchronize()

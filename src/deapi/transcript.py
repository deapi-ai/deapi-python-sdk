"""Transcript result parsing for v2 transcription jobs.

A transcription job's result comes in one of two shapes, decided by the model
card in ``GET /models`` (``info.limits.output_format``):

* ``"json"`` (e.g. ``WhisperLargeV3Ct2``): the transcript is always a JSON
  document ``{"text", "language", "ts_level", "diarization_available",
  "segments": [...]}`` -- even with timestamps off.
* ``null`` (e.g. ``WhisperLargeV3``): plain text, with timestamps written into
  the text itself (``[0:50 - 1:49] ...``) when requested.

``parse_transcript`` turns either into one ``Transcript`` object.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TranscriptWord(BaseModel):
    """One word with timing. Extra fields (e.g. ``chars``) are kept."""

    model_config = ConfigDict(extra="allow")

    word: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[str] = None
    score: Optional[float] = None


class TranscriptSegment(BaseModel):
    """One timed segment. ``speaker`` is set when diarization ran, ``words`` at word level."""

    model_config = ConfigDict(extra="allow")

    start: Optional[float] = None
    end: Optional[float] = None
    text: str = ""
    speaker: Optional[str] = None
    words: Optional[List[TranscriptWord]] = None


class Transcript(BaseModel):
    """A parsed transcription result.

    ``format`` is ``"json"`` for a structured document and ``"text"`` for plain
    text (``segments`` is then empty and ``language`` / ``ts_level`` are None).
    ``raw`` is always the untouched result string.
    """

    model_config = ConfigDict(extra="allow")

    text: str
    format: Literal["json", "text"]
    language: Optional[str] = None
    ts_level: Optional[str] = None
    diarization_available: Optional[bool] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)
    raw: str = ""


def transcript_output_format(model_card: Any) -> Optional[str]:
    """Read ``info.limits.output_format`` from a model card (dict or pydantic model).

    Returns ``"json"`` or None (None means the model returns plain text).
    """
    if hasattr(model_card, "model_dump"):
        model_card = model_card.model_dump()
    if not isinstance(model_card, dict):
        return None
    info = model_card.get("info")
    limits = info.get("limits") if isinstance(info, dict) else None
    value = limits.get("output_format") if isinstance(limits, dict) else None
    return value if isinstance(value, str) and value else None


def _decode_object(raw: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the transcript JSON object, or None if ``raw`` is not one."""
    if isinstance(raw, dict):
        doc: Any = raw
    else:
        if not raw.lstrip().startswith("{"):
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
        return None
    return doc


def parse_transcript(
    raw: Union[str, Dict[str, Any], None],
    *,
    output_format: Optional[str] = None,
) -> Transcript:
    """Parse a transcription result (``JobResult.result`` or a downloaded file).

    Args:
        raw: The result string, or an already-decoded JSON object.
        output_format: The model card's ``info.limits.output_format``.
            ``"json"`` or None (auto): a JSON object with a ``text`` field is
            parsed as a structured transcript. ``"text"``: never decode, the
            string is returned as is.

    Never raises on malformed or unexpected input -- anything that is not a
    structured transcript comes back as ``format="text"``.
    """
    if raw is None:
        return Transcript(text="", format="text", raw="")
    raw_str = raw if isinstance(raw, str) else json.dumps(raw)

    doc = None if output_format == "text" else _decode_object(raw)
    if doc is None:
        return Transcript(text=raw_str, format="text", raw=raw_str)

    segments: List[TranscriptSegment] = []
    raw_segments = doc.get("segments")
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            try:
                segments.append(TranscriptSegment.model_validate(seg))
            except Exception:
                continue

    diarization = doc.get("diarization_available")
    language = doc.get("language")
    ts_level = doc.get("ts_level")
    return Transcript(
        text=doc["text"],
        format="json",
        language=language if isinstance(language, str) else None,
        ts_level=ts_level if isinstance(ts_level, str) else None,
        diarization_available=diarization if isinstance(diarization, bool) else None,
        segments=segments,
        raw=raw_str,
    )

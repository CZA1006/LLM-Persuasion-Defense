# src/telemetry.py
# -*- coding: utf-8 -*-
"""
Lightweight JSONL tracer for experiments.

- Backward compatible with existing usage in this repo.
- Adds optional fields for X-Teaming and cost tracking:
  plans, draft_scores, chosen_idx, diagnosis, rewrite_used, usage, latency_ms.
- Safe to disable (enabled=False) -> methods become no-ops.
- NEW:
  * autostart: automatically writes a 'run_start' header on init (default True)
  * log(): backward-compat alias for ad-hoc dict logging (keeps old code working)
  * robust file naming + thread-safe appends
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

LOGGER = logging.getLogger("telemetry")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


def _now_iso() -> str:
    """ISO8601 timestamp with seconds precision (local time)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class TraceWriter:
    """
    JSONL trace writer.

    Typical usage:
        tw = TraceWriter(trace_dir="traces", tag="oai_full_turns_noinj_v1",
                         enabled=True, run_meta={...})  # autostart=True by default
        tw.write_meta(turn=1, turn_strategy="flattery", strategy_regime="crescendo")
        tw.write_turn(turn=1, strategy="flattery", user_compiled="...", user_injected="...",
                      answer="...", hit=True, plans=[...], draft_scores=[...], chosen_idx=0,
                      diagnosis={...}, rewrite_used=False, usage={"total_tokens": 123}, latency_ms=456)
        tw.write_summary({"PSR": 0.8, "N": 60})
        tw.close()

    All write_* methods are no-ops when enabled=False.
    """
    trace_dir: Union[Path, str] = "traces"
    tag: Optional[str] = None
    enabled: bool = True
    run_meta: Optional[Dict[str, Any]] = None
    autostart: bool = True  # NEW: write run_start automatically

    _path: Optional[Path] = field(default=None, init=False, repr=False)
    _fh: Optional[Any] = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _opened: bool = field(default=False, init=False, repr=False)
    _header_written: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        trace_dir = Path(self.trace_dir)
        _ensure_dir(trace_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"trace_{ts}"
        if self.tag:
            # Sanitize tag to avoid weird filesystem chars
            safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in self.tag)
            fname += f"_{safe_tag}"
        fname += ".jsonl"
        self._path = trace_dir / fname

        # Auto write run_start header once
        if self.autostart:
            self.write_run_header()

    # -------------------------
    # Internal
    # -------------------------

    def _open_if_needed(self) -> None:
        if not self.enabled:
            return
        if self._opened:
            return
        assert self._path is not None, "TraceWriter path is not initialized"
        # append mode to allow additional events if process restarts/resumes
        self._fh = self._path.open("a", encoding="utf-8")
        self._opened = True
        LOGGER.info("Trace file: %s", str(self._path))

    def _write_event(self, obj: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._open_if_needed()
            assert self._fh is not None
            line = json.dumps(obj, ensure_ascii=False)
            self._fh.write(line + "\n")
            self._fh.flush()

    # -------------------------
    # Public API
    # -------------------------

    @property
    def path(self) -> Optional[Path]:
        """Return the file path (if enabled); else None."""
        return self._path if self.enabled else None

    def write_run_header(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Write the run_start event with run_meta and optional extra fields.
        Guarded to only write once.
        """
        if not self.enabled or self._header_written:
            return
        payload: Dict[str, Any] = {
            "_event": "run_start",
            "ts": _now_iso(),
        }
        if self.run_meta:
            payload["run_meta"] = self.run_meta
        if extra:
            payload.update(extra)
        self._write_event(payload)
        self._header_written = True

    def write_meta(self, **kwargs: Any) -> None:
        """
        Generic metadata event. Useful to log per-turn planning info,
        chosen strategy, etc. Example:
            write_meta(turn=3, turn_strategy="authority", strategy_regime="crescendo")
        """
        if not self.enabled:
            return
        payload: Dict[str, Any] = {"_event": "meta", "ts": _now_iso()}
        payload.update(kwargs)
        self._write_event(payload)

    def write_turn(
        self,
        *,
        turn: int,
        strategy: str,
        user_compiled: str,
        user_injected: str,
        answer: str,
        hit: bool,
        # Optional known fields (all are optional & backward compatible)
        provider: Optional[str] = None,
        model: Optional[str] = None,
        defense: Optional[str] = None,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        o_true: Optional[str] = None,
        o_false: Optional[str] = None,
        # X-Teaming / analysis extras:
        plans: Optional[Any] = None,
        draft_scores: Optional[Any] = None,
        chosen_idx: Optional[int] = None,
        diagnosis: Optional[Any] = None,
        rewrite_used: bool = False,
        # Cost & timing:
        usage: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
        # Anything else:
        **extra: Any,
    ) -> None:
        """
        Write one turn record. Unknown kwargs are merged into the record.
        """
        if not self.enabled:
            return
        rec: Dict[str, Any] = {
            "_event": "turn",
            "ts": _now_iso(),
            "turn": turn,
            "strategy": strategy,
            "user_compiled": user_compiled,
            "user_injected": user_injected,
            "answer": answer,
            "hit_o_false": bool(hit),
        }
        if provider is not None: rec["provider"] = provider
        if model is not None: rec["model"] = model
        if defense is not None: rec["defense"] = defense
        if subject is not None: rec["subject"] = subject
        if relation is not None: rec["relation"] = relation
        if o_true is not None: rec["o_true"] = o_true
        if o_false is not None: rec["o_false"] = o_false

        # Optional, only included when provided
        if plans is not None: rec["plans"] = plans
        if draft_scores is not None: rec["draft_scores"] = draft_scores
        if chosen_idx is not None: rec["chosen_idx"] = chosen_idx
        if diagnosis is not None: rec["diagnosis"] = diagnosis
        if rewrite_used: rec["rewrite_used"] = True
        if usage is not None: rec["usage"] = usage
        if latency_ms is not None: rec["latency_ms"] = latency_ms

        # Merge extra unknown fields for forward-compatibility
        if extra:
            for k, v in extra.items():
                # avoid silent override for core keys
                if k in rec:
                    LOGGER.debug("write_turn: overriding key '%s'", k)
                rec[k] = v

        self._write_event(rec)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        """
        Write a summary event at the end of a run.
        """
        if not self.enabled:
            return
        payload = {
            "_event": "summary",
            "ts": _now_iso(),
        }
        payload.update(summary)
        self._write_event(payload)

    def write_run_end(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the end of the run. Optional extra fields.
        """
        if not self.enabled:
            return
        payload: Dict[str, Any] = {"_event": "run_end", "ts": _now_iso()}
        if extra:
            payload.update(extra)
        self._write_event(payload)

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self.enabled:
            return
        with self._lock:
            try:
                if self._fh:
                    self._fh.flush()
                    self._fh.close()
            except Exception:
                pass
            finally:
                self._fh = None
                self._opened = False

    # -------------------------
    # Backward-compat alias
    # -------------------------
    def log(self, obj: Dict[str, Any]) -> None:
        """
        Backward-compat: write arbitrary dict as a JSONL event (old code calls tracer.log).
        Ensures _event + ts present; defaults to '_event'='turn'.
        """
        if not self.enabled:
            return
        rec = dict(obj) if isinstance(obj, dict) else {"payload": str(obj)}
        rec.setdefault("_event", "turn")
        rec.setdefault("ts", _now_iso())
        self._write_event(rec)


# -------------------------
# Convenience factory
# -------------------------

def make_trace_writer(
    enabled: bool,
    trace_dir: Union[str, Path] = "traces",
    tag: Optional[str] = None,
    run_meta: Optional[Dict[str, Any]] = None,
    autostart: bool = True,
) -> TraceWriter:
    """
    Create a TraceWriter with optional run_meta and (by default) write the run_start header.
    """
    tw = TraceWriter(trace_dir=trace_dir, tag=tag, enabled=enabled, run_meta=run_meta, autostart=autostart)
    return tw

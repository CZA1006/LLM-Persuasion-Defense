# src/telemetry.py
from __future__ import annotations
import json, os, time, uuid
from pathlib import Path
from typing import Optional, Dict, Any

class TraceWriter:
    """
    Append-only JSONL tracer. Each call to .log(record) writes one line.
    Safe to pass None elsewhere (no-op).
    """
    def __init__(self, trace_dir: str = "traces", run_tag: Optional[str] = None):
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"trace_{tag}_{uuid.uuid4().hex[:8]}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        # minimal header record
        self.log({"_event": "trace_open", "path": str(self.path), "ts": time.time()})

    def log(self, record: Dict[str, Any]) -> None:
        if not record:
            return
        record = dict(record)
        record.setdefault("ts", time.time())
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self.log({"_event": "trace_close", "ts": time.time()})
            self._fh.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

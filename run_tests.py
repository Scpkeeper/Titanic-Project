#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
stream = io.StringIO()
suite = unittest.defaultTestLoader.discover(
    str(ROOT),
    pattern="test_titanic_b_pipeline.py",
)
result = unittest.TextTestRunner(
    stream=stream,
    verbosity=2,
).run(suite)
text = stream.getvalue()
print(text)

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
    "successful": result.wasSuccessful(),
    "output": text,
}
reports_dir = ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
(reports_dir / "test_results.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
sys.exit(0 if result.wasSuccessful() else 1)

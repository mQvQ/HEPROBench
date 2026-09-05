from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("HEPROBENCH_RUN_E2E") == "1",
    "set HEPROBENCH_RUN_E2E=1 to run native train-infer-evaluate demos",
)
class NativeEndToEndTests(unittest.TestCase):
    def test_all_method_demos(self) -> None:
        device = os.environ.get("HEPROBENCH_E2E_DEVICE", "cuda:0")
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "verification.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_all_method_demos.py"),
                    "--device",
                    device,
                    "--summary",
                    str(summary_path),
                ],
                cwd=ROOT,
                check=True,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["schema"], "heprobench_demo_verification_v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["passed_methods"], 10)
        self.assertEqual(summary["total_methods"], 10)


if __name__ == "__main__":
    unittest.main()

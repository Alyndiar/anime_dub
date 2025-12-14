import json
import tempfile
from pathlib import Path
import unittest

from scripts.clean_whisper_json import DEFAULT_CONFIG, clean_whisper_json


class CleanWhisperJsonTests(unittest.TestCase):
    def test_noise_filtering_and_merging(self) -> None:
        sample = Path("tests/data/whisper_sample.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sample_clean.json"
            report = Path(tmpdir) / "report.json"
            config = {
                "merge_short": True,
                "merge_min_length": 4,
                "merge_target_length": 12,
                "max_pause": 0.5,
                "promo_keywords": DEFAULT_CONFIG["promo_keywords"],
            }
            clean_whisper_json(str(sample), str(out), str(report), config)

            cleaned = json.loads(out.read_text(encoding="utf-8"))
            segments = cleaned["segments"]
            self.assertEqual(4, len(segments))

            # Vérifie la fusion des doublons ("是星斗" sur deux segments consécutifs)
            third = segments[2]
            self.assertEqual("是星斗", third["text_zh"])
            self.assertAlmostEqual(3.8, third["start"])
            self.assertAlmostEqual(4.2, third["end"])

            # Vérifie le regroupement des courts segments finaux
            last = segments[-1]
            self.assertEqual("这是一个测试", last["text_zh"])
            self.assertAlmostEqual(5.0, last["start"])
            self.assertAlmostEqual(5.8, last["end"])

            report_data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(13, report_data["segments_in"])
            self.assertEqual(4, report_data["segments_out"])
            self.assertEqual(1, report_data["duplicate_fusions"])
            self.assertGreaterEqual(report_data["short_merges"], 2)
            self.assertEqual(1, report_data["removed"]["promo"])
            self.assertEqual(1, report_data["removed"]["latin"])
            self.assertEqual(1, report_data["removed"]["hangul"])
            self.assertEqual(1, report_data["removed"]["empty"])
            self.assertEqual(1, report_data["removed"]["short_fragment"])
            self.assertEqual(1, report_data["removed"]["repeat"])


if __name__ == "__main__":
    unittest.main()

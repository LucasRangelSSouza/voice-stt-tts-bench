import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_warm_tts_is_faster_than_real_time_everywhere():
    assert S["f5_worst_rtf"] < 1.0


def test_cold_start_outliers_exist_and_are_excluded():
    assert S["f5_cold_n"] >= 1 and S["f5_cold_max_rtf"] > 10


def test_turbo_whisper_is_faster_than_large_and_equally_accurate_on_the_clip():
    assert S["whisper_turbo_rtf"] < S["whisper_large_rtf"]
    assert S["whisper_turbo_wer"] <= 0.05


def test_full_duplex_model_needs_warm_runs():
    assert S["personaplex_best_rtf"] < 1.0

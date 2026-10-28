# Data

Curated CSV files from the campaign (public Kaggle dataset `lucasrangelss/gpu-ai-benchmark-campaign-2026`). One row per measured run or attempt; the columns are described in the dataset page. Host locations and instance identifiers were removed.

- `libs_runs.csv`: every timed run of the per-card library sweeps (F5-TTS and faster-whisper rows are used here)
- `sandbox_metrics.csv`: metrics printed by the sandbox tests (faster-whisper model sizes on an RTX 3090, among others)
- `openvoice_cpu_local.csv`: OpenVoice v2 with MeloTTS on a laptop CPU in Docker
- `media_gated_runs.csv`: FLUX.3 Klein and PersonaPlex runs on the gated-weights sessions (PersonaPlex rows are used here)

#!/bin/bash
# Session for models with gated weights: PersonaPlex full-duplex voice and FLUX.3 Klein.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
# FLUX.3 Klein (CLI oficial) e NVIDIA PersonaPlex (offline) com HF_TOKEN no ambiente. Cada um em venv proprio.
cd /workspace/eco; mkdir -p out/gated /workspace/g; cd /workspace/g
pip install -q uv >/dev/null 2>&1
log() { echo "{\"lib\":\"$1\",\"status\":\"$2\",\"note\":\"$3\"}" >> /workspace/eco/out/gated/gated.jsonl; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee /workspace/eco/out/gated/gpu.txt
# ---------------- FLUX.3 Klein
T0=$(date +%s)
git clone -q --depth 1 https://github.com/black-forest-labs/flux-3-image-klein.git flux && cd flux
uv venv .venv --python 3.12 -q; . .venv/bin/activate
(uv pip install -q -e . --extra-index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2); log flux install "$(( $(date +%s)-T0 ))s"
T1=$(date +%s)
hf download black-forest-labs/flux-3-image-klein --local-dir weights/flux-3-image-klein --exclude 'variants/*' > /workspace/eco/out/gated/flux_dl1.log 2>&1
hf download black-forest-labs/flux-3-image-klein --local-dir weights/flux-3-image-klein --include 'variants/fp8r/*' > /workspace/eco/out/gated/flux_dl1b.log 2>&1
hf download black-forest-labs/flux-3-image-klein-pup --local-dir weights/pup > /workspace/eco/out/gated/flux_dl2.log 2>&1
hf download Falconsai/nsfw_image_detection --local-dir weights/nsfw-classifier > /workspace/eco/out/gated/flux_dl3.log 2>&1
log flux download "$(( $(date +%s)-T1 ))s; $(du -sh weights | cut -f1)"
cat > /workspace/g/flux_bench.sh <<'SH'
cd /workspace/g/flux; . .venv/bin/activate
for P in "A cozy artisan bakery at dawn, warm light, bread on wooden shelves" "A man walking with his dog, afternoon sun" "Isometric illustration of a small data center with glowing servers"; do
  for FP in "" "--fp8"; do
    S=$(date +%s.%N)
    python scripts/cli.py "$P" --model distilled $FP --out /workspace/eco/out/gated/flux_$(date +%s).png --seed 7 2>&1 | tail -40
    E=$(date +%s.%N); echo "METRIC_FLUX fp=${FP:-bf16} seconds=$(echo "$E - $S" | bc)"
  done
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
SH
timeout 2400 bash /workspace/g/flux_bench.sh > /workspace/eco/out/gated/flux_result.txt 2>&1
log flux bench "$(grep METRIC_FLUX /workspace/eco/out/gated/flux_result.txt | tr '\n' ' ' | cut -c1-400)"
deactivate; cd /workspace/g
# ---------------- PersonaPlex
T0=$(date +%s)
git clone -q --depth 1 https://github.com/NVIDIA/personaplex.git pp && cd pp
uv venv .venv --python 3.12 -q; . .venv/bin/activate
uv pip install -q moshi/. 2>&1 | tail -1; uv pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
log personaplex install "$(( $(date +%s)-T0 ))s"
ls assets/test 2>/dev/null | head -5
T1=$(date +%s)
for V in NATF2 NATM1; do
  S=$(date +%s.%N)
  python -m moshi.offline --voice-prompt "$V.pt" --input-wav assets/test/input_assistant.wav --seed 42424242 --output-wav /workspace/eco/out/gated/pp_$V.wav --output-text /workspace/eco/out/gated/pp_$V.json > /workspace/eco/out/gated/pp_$V.log 2>&1
  E=$(date +%s.%N)
  python - <<PY
import wave
try:
    w=wave.open("/workspace/eco/out/gated/pp_$V.wav"); print("METRIC_PP voice=$V audio_s=%.1f wall_s=%.1f" % (w.getnframes()/w.getframerate(), $E-$S))
except Exception as e: print("METRIC_PP voice=$V error", repr(e)[:120])
PY
done > /workspace/eco/out/gated/pp_result.txt 2>&1
log personaplex bench "$(cat /workspace/eco/out/gated/pp_result.txt | tr '\n' ' ' | cut -c1-400) total $(( $(date +%s)-T1 ))s"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo SCRIPT_DONE

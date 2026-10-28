# GPU sandbox suite: faster-whisper model sizes with WER, OpenVoice, ComfyUI, llama.cpp.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
"""Testes de bibliotecas que dependem de GPU (cada uma em venv proprio, timeout, metricas). Reusa o runner do sandbox_suite.
uso: python gpu_suite.py [ids...]"""
import sys

sys.path.insert(0, "/workspace/eco")
import sandbox_suite as S  # noqa: E402

G = {}


def g(id_, name, script, timeout=1800):
    G[id_] = dict(name=name, script=script, timeout=timeout, tier="T2-GPU")


PRE = """export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nvcc --version | tail -1
pip install -q uv >/dev/null 2>&1
"""

g("gpu_faster_whisper", "faster-whisper", PRE + """uv venv v --python 3.11 -q; . v/bin/activate
uv pip install -q faster-whisper "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*" jiwer soundfile numpy 2>&1 | tail -1
export LD_LIBRARY_PATH=$(python - <<'PY'
import nvidia.cublas.lib, nvidia.cudnn.lib, os
print(os.path.dirname(nvidia.cublas.lib.__file__)+":"+os.path.dirname(nvidia.cudnn.lib.__file__))
PY
):$LD_LIBRARY_PATH
curl -sLo jfk.wav https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
python - <<'PY'
import time, numpy as np, soundfile as sf, jiwer
from faster_whisper import WhisperModel
a, sr = sf.read("jfk.wav", dtype="float32")
long = np.concatenate([a] * 6); sf.write("long.wav", long, sr)
ref = " ".join(["and so my fellow americans ask not what your country can do for you ask what you can do for your country"] * 6)
for sz in ("small", "large-v3-turbo", "large-v3"):
    m = WhisperModel(sz, device="cuda", compute_type="float16")
    m.transcribe("jfk.wav", language="en")
    t0 = time.time(); segs, info = m.transcribe("long.wav", language="en", beam_size=1); hyp = " ".join(s.text for s in segs); dt = time.time() - t0
    hyp = "".join(c for c in hyp.lower() if c.isalpha() or c == " ")
    print("METRIC %s_rtf=%.4f %s_wer=%.3f audio_s=%.1f" % (sz.replace("-", "_"), dt / info.duration, sz.replace("-", "_"), jiwer.wer(ref, hyp), info.duration), flush=True)
PY""", 1500)

g("gpu_detectron2", "Detectron2", PRE + """uv venv d --python 3.10 -q; . d/bin/activate
uv pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
uv pip install -q "setuptools<81" wheel ninja opencv-python-headless pillow numpy 2>&1 | tail -1
(time CUDA_HOME=/usr/local/cuda uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git" 2>&1 | tail -3) 2>&1 | tail -6
mkdir -p img; for i in 000000039769 000000000139 000000000285 000000000632 000000000724 000000000776; do curl -sL -o img/$i.jpg http://images.cocodataset.org/val2017/$i.jpg; done
python - <<'PY'
import time, glob, cv2, torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
for name in ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"):
    cfg = get_cfg(); cfg.merge_from_file(model_zoo.get_config_file(name)); cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name); cfg.MODEL.DEVICE = "cuda"
    p = DefaultPredictor(cfg); imgs = [cv2.imread(f) for f in sorted(glob.glob("img/*.jpg"))]
    for im in imgs[:2]: p(im)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); ts = []
    for i in range(30):
        t0 = time.time(); p(imgs[i % len(imgs)]); torch.cuda.synchronize(); ts.append(time.time() - t0)
    ts.sort(); print("METRIC %s_p50_ms=%.1f %s_p95_ms=%.1f %s_vram_gb=%.2f" % (name.split("/")[1][:12].replace("-", "_"), ts[15] * 1000, name.split("/")[1][:12].replace("-", "_"), ts[28] * 1000, name.split("/")[1][:12].replace("-", "_"), torch.cuda.max_memory_allocated() / 2**30))
PY""", 2400)

g("gpu_comfyui", "ComfyUI", PRE + """uv venv c --python 3.11 -q; . c/bin/activate
git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI && cd ComfyUI
uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1
uv pip install -q -r requirements.txt 2>&1 | tail -1
mkdir -p models/checkpoints; hf download stabilityai/sdxl-turbo sd_xl_turbo_1.0_fp16.safetensors --local-dir models/checkpoints >/dev/null 2>&1; ls -la models/checkpoints | tail -2
(nohup python main.py --listen 127.0.0.1 --port 8188 > comfy.log 2>&1 &)
for i in $(seq 1 60); do sleep 3; curl -s http://127.0.0.1:8188/system_stats >/dev/null && { echo "METRIC ready_s=$((i*3))"; break; }; done
python - <<'PY'
import json, time, urllib.request, random
def wf(seed, bs):
    return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "foto de uma padaria artesanal ao amanhecer, luz suave, alta qualidade", "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": bs}},
            "5": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler_ancestral", "scheduler": "simple", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "bench"}}}
def run(seed, bs):
    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": wf(seed, bs)}).encode(), headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    while True:
        h = json.load(urllib.request.urlopen("http://127.0.0.1:8188/history/" + pid))
        if pid in h: break
        time.sleep(0.2)
    return time.time() - t0
cold = run(1, 1); print("METRIC first_call_s=%.2f" % cold)
for bs in (1, 4):
    ts = sorted(run(10 + i, bs) for i in range(5)); print("METRIC bs%d_s_per_batch_p50=%.3f bs%d_s_per_image=%.3f" % (bs, ts[2], bs, ts[2] / bs))
PY
nvidia-smi --query-gpu=memory.used --format=csv,noheader""", 2400)

g("gpu_llamacpp", "llama.cpp", PRE + """git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git lc && cd lc
(time cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null 2>&1 && cmake --build build -j16 --target llama-bench >/dev/null 2>&1) 2>&1 | tail -3
pip install -q -U "huggingface_hub[cli]" >/dev/null 2>&1
python -c "from huggingface_hub import hf_hub_download as h; [h('unsloth/Qwen3.5-9B-GGUF', f, local_dir='.') for f in ('Qwen3.5-9B-Q4_K_M.gguf','Qwen3.5-9B-Q8_0.gguf')]" 2>&1 | tail -2; ls *.gguf
for f in *.gguf; do echo "== $f"; ./build/bin/llama-bench -m "$f" -ngl 99 -p 512 -n 128 -r 3 2>&1 | grep -E "pp512|tg128" | sed 's/|/ /g'; done""", 2700)

PY9 = """hf download Qwen/Qwen3.5-9B --local-dir /workspace/models/q9 >/dev/null 2>&1; echo dl_ok"""
BENCH = """for L in 1 8 16; do :; done
python /workspace/eco/llm_bench.py --base http://127.0.0.1:18000/v1 --model q9 --out ENGINE_OUT --levels 1,8,16 --dur 25 --think off 2>&1 | tail -4"""

g("gpu_engine_vllm", "vLLM (9B, comparacao de engine)", PRE + PY9 + """
. /venv/main/bin/activate; pip install -q httpx 2>&1 | tail -1; python -c "import vllm;print('vllm',vllm.__version__)"
(nohup vllm serve /workspace/models/q9 --served-model-name q9 --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 16 --reasoning-parser qwen3 > vllm.log 2>&1 &)
for i in $(seq 1 120); do sleep 5; [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:18000/health)" = 200 ] && { echo "METRIC ready_s=$((i*5))"; break; }; done
""" + BENCH.replace("ENGINE_OUT", "/workspace/eco/out/engine_vllm") + """
tail -40 vllm.log | cut -c1-240; cp vllm.log /workspace/eco/out/engine_vllm_server.log; pkill -f "[v]llm serve"; sleep 5""", 2700)

g("gpu_engine_sglang", "SGLang (9B)", PRE + PY9 + """
uv venv se --python 3.12 -q; . se/bin/activate
(time uv pip install -q "sglang[all]" httpx 2>&1 | tail -2) 2>&1 | tail -4
(nohup python -m sglang.launch_server --model-path /workspace/models/q9 --served-model-name q9 --host 127.0.0.1 --port 18000 --mem-fraction-static 0.85 --context-length 8192 --reasoning-parser qwen3 > sglang.log 2>&1 &)
for i in $(seq 1 120); do sleep 5; [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:18000/health)" = 200 ] && { echo "METRIC ready_s=$((i*5))"; break; }; done
tail -3 sglang.log | cut -c1-200
""" + BENCH.replace("ENGINE_OUT", "/workspace/eco/out/engine_sglang") + """
cp sglang.log /workspace/eco/out/engine_sglang_server.log; pkill -f "[s]glang.launch"; sleep 5""", 2700)

g("gpu_openvoice", "OpenVoice", S.T["openvoice_cpu"]["script"].replace('dev="cpu"', 'dev="cuda"').replace('map_location=dev', 'map_location=dev'), 2400)

g("gpu_vtuber", "Open-LLM-VTuber", PRE + """git clone -q --depth 1 https://github.com/Open-LLM-VTuber/Open-LLM-VTuber.git vt && cd vt
(time uv sync >sync.log 2>&1) 2>&1 | tail -3; tail -2 sync.log | cut -c1-200; du -sh .venv | tail -1
ls; ls config_templates 2>/dev/null | head; cp config_templates/conf.default.yaml conf.yaml 2>/dev/null || cp conf.yaml.template conf.yaml 2>/dev/null
grep -n -E "llm_provider|asr_model|tts_model|ollama|faster_whisper|edge_tts" conf.yaml | head -20
(nohup uv run run_server.py > server.log 2>&1 &)
for i in $(seq 1 60); do sleep 5; curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:12393/ | grep -q 200 && { echo "METRIC server_ready_s=$((i*5))"; break; }; done
tail -15 server.log | cut -c1-220
ps -eo rss,cmd | grep -i run_server | sort -rn | head -1 | cut -c1-120
nvidia-smi --query-gpu=memory.used --format=csv,noheader""", 1800)

g("gpu_supra_tex", "SUPRA-Tex", """echo "NOTE: pipeline imagem->3D exige DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m), repositorio gated no HuggingFace: precisa de token com acesso aprovado" """, 60)

# bateria multi-GPU do 9B: velocidade + custo
g("gpu_9b_speed", "Qwen3.5-9B (velocidade por GPU)", PRE + PY9 + """
. /venv/main/bin/activate 2>/dev/null || true
(nohup vllm serve /workspace/models/q9 --served-model-name q9 --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 32 --reasoning-parser qwen3 > vllm9.log 2>&1 &)
for i in $(seq 1 120); do sleep 5; [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:18000/health)" = 200 ] && { echo "METRIC ready_s=$((i*5))"; break; }; done
python /workspace/eco/llm_bench.py --base http://127.0.0.1:18000/v1 --model q9 --out /workspace/eco/out/b9speed --levels 1,4,16,32 --dur 25 --think off 2>&1 | tail -4
pkill -f "[v]llm serve" """, 2400)

if __name__ == "__main__":
    ids = sys.argv[1:] or list(G)
    for i in ids:
        try:
            S.run_test(i, G[i])
        except Exception as e:  # noqa: BLE001
            import json
            open(S.RES, "a").write(json.dumps({"id": i, "status": "runner_error", "err": repr(e)[:300]}) + chr(10))
    print("SUITE_DONE", flush=True)

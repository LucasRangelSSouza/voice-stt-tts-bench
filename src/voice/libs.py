# Per-library timing functions, including F5-TTS and faster-whisper. One JSON line per run.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
"""Bench das bibliotecas que dependem de GPU. Cada subcomando instala (se preciso), mede e escreve 1 JSON por linha em OUT/libs.jsonl.
Falha/incompatibilidade vira registro failed/incompatible com o log; nunca vira numero inventado."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

OUT = os.environ.get("OUT", "/workspace/eco/out/libs")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "libs.jsonl")
PY = sys.executable


def rec(lib, status, **kw):
    kw.update({"lib": lib, "status": status, "t_wall": time.time()})
    try:
        kw["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    open(LOG, "a").write(json.dumps(kw, default=str) + "\n")
    print(lib, status, {k: v for k, v in kw.items() if k not in ("t_wall", "gpu")}, flush=True)


def sh(cmd, timeout=900):
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-1500:], time.time() - t0


def pip(pkgs, timeout=900):
    return sh(f"{PY} -m pip install -q {pkgs}", timeout)


def vram_peak():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def timed(fn, n, warm=3):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return {"n": n, "p50_ms": round(ts[len(ts) // 2] * 1000, 2), "p95_ms": round(ts[max(0, int(len(ts) * .95) - 1)] * 1000, 2),
            "mean_ms": round(sum(ts) / len(ts) * 1000, 2)}


def sample_images():
    d = "/workspace/eco/data/img"
    os.makedirs(d, exist_ok=True)
    if len(glob.glob(d + "/*.jpg")) < 8:
        for i in ["000000039769", "000000000139", "000000000285", "000000000632", "000000000724", "000000000776", "000000000785",
                  "000000000802"]:
            sh(f"curl -sL -o {d}/{i}.jpg http://images.cocodataset.org/val2017/{i}.jpg", 60)
    return sorted(glob.glob(d + "/*.jpg"))


def yolo():
    pip("ultralytics")
    import numpy as np
    import torch
    from ultralytics import YOLO
    imgs = sample_images()
    for name in ("yolo11n", "yolo11s", "yolo11m", "yolo11x"):
        try:
            m = YOLO(f"{name}.pt")
            for bs in (1, 8):
                torch.cuda.reset_peak_memory_stats()
                batch = (imgs * 4)[:bs]
                r = timed(lambda: m.predict(batch, imgsz=640, half=True, verbose=False, device=0), 20)
                rec("yolo", "ok", model=name, imgsz=640, half=True, batch=bs, **r, fps_effective=round(bs * 1000 / r["p50_ms"], 1),
                    vram_gb=vram_peak(), measurement_kind="measured")
            fr = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            r = timed(lambda: m.predict(fr, imgsz=1280, half=True, verbose=False, device=0), 20)
            rec("yolo", "ok", model=name, imgsz=1280, input="1080p", batch=1, **r, measurement_kind="measured")
        except Exception as e:  # noqa: BLE001
            rec("yolo", "failed", model=name, err=repr(e)[:300], tb=traceback.format_exc()[-600:])


def track():
    """ByteTrack e BoT-SORT via ultralytics sobre video curto: fps do pipeline detector+tracker."""
    pip("ultralytics lap")
    from ultralytics import YOLO
    vid = "/workspace/eco/data/clip.mp4"
    if not os.path.exists(vid):
        sh(f"curl -sL -o {vid} https://github.com/ultralytics/assets/releases/download/v0.0.0/decelera_landscape_min.mov", 120)
    if not os.path.exists(vid) or os.path.getsize(vid) < 100000:
        return rec("track", "skipped", reason="sem video de teste (download falhou)")
    for tr in ("bytetrack.yaml", "botsort.yaml"):
        try:
            m = YOLO("yolo11m.pt")
            t0 = time.time()
            n = 0
            ids = set()
            for r in m.track(vid, tracker=tr, stream=True, verbose=False, half=True, imgsz=640, persist=True):
                n += 1
                if r.boxes.id is not None:
                    ids.update(r.boxes.id.int().tolist())
                if n >= 600:
                    break
            dt = time.time() - t0
            rec("track", "ok", tracker=tr, frames=n, wall_s=round(dt, 2), fps=round(n / dt, 1), unique_ids=len(ids),
                note="sem ground truth: HOTA/IDF1 NOT_TESTED", measurement_kind="measured")
        except Exception as e:  # noqa: BLE001
            rec("track", "failed", tracker=tr, err=repr(e)[:300])


def yolo2():
    """Tempo so de GPU (tensor ja em GPU, sem pre/pos-processamento de CPU) + tracker com 600 frames do video em loop."""
    os.makedirs("/workspace/eco/data", exist_ok=True)
    open("/workspace/eco/data/.extra_done", "w").write("1")
    pip("ultralytics lap opencv-python-headless")
    import torch
    from ultralytics import YOLO
    for name in ("yolo11n", "yolo11m", "yolo11x"):
        try:
            m = YOLO(f"{name}.pt")
            net = m.model.cuda().half().eval()
            for bs, sz in ((1, 640), (8, 640), (16, 640), (1, 1280), (4, 1280)):
                x = torch.rand(bs, 3, sz, sz, device="cuda", dtype=torch.float16)
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    r = timed(lambda: net(x), 30, warm=5)
                rec("yolo_gpu_only", "ok", model=name, batch=bs, imgsz=sz, **r, fps_gpu=round(bs * 1000 / r["p50_ms"], 1), vram_gb=vram_peak(),
                    measurement_kind="measured")
        except Exception as e:  # noqa: BLE001
            rec("yolo_gpu_only", "failed", model=name, err=repr(e)[:300])


def track2():
    import cv2
    from ultralytics import YOLO
    vid = "/workspace/eco/data/clip.mp4"
    if not os.path.exists(vid):
        sh(f"curl -sL -o {vid} https://github.com/ultralytics/assets/releases/download/v0.0.0/decelera_landscape_min.mov", 120)
    cap = cv2.VideoCapture(vid)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        return rec("track2", "skipped", reason="video ilegivel")
    for tr in ("bytetrack.yaml", "botsort.yaml"):
        for mname in ("yolo11n", "yolo11m"):
            try:
                m = YOLO(f"{mname}.pt")
                for i in range(15):
                    m.track(frames[i % len(frames)], tracker=tr, persist=True, verbose=False, half=True)
                m = YOLO(f"{mname}.pt")
                t0 = time.time()
                ids = set()
                n = 600
                for i in range(n):
                    r = m.track(frames[i % len(frames)], tracker=tr, persist=True, verbose=False, half=True, imgsz=640)[0]
                    if r.boxes.id is not None:
                        ids.update(r.boxes.id.int().tolist())
                dt = time.time() - t0
                rec("track", "ok", tracker=tr, detector=mname, frames=n, src_frames=len(frames), wall_s=round(dt, 2), fps=round(n / dt, 1),
                    unique_ids=len(ids), note="sem ground truth: HOTA/IDF1 NOT_TESTED", measurement_kind="measured")
            except Exception as e:  # noqa: BLE001
                rec("track", "failed", tracker=tr, detector=mname, err=repr(e)[:300])


def sam2():
    rc, o, t = pip("git+https://github.com/facebookresearch/sam2.git", 900)
    if rc:
        return rec("sam2", "failed", stage="install", log=o)
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    ck = "/workspace/eco/data/sam2.1_hiera_large.pt"
    if not os.path.exists(ck):
        sh(f"curl -sL -o {ck} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt", 600)
    try:
        torch.cuda.reset_peak_memory_stats()
        pred = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ck, device="cuda"))
        from PIL import Image
        img = np.array(Image.open(sample_images()[0]).convert("RGB"))

        def f():
            pred.set_image(img)
            pred.predict(point_coords=np.array([[200, 200]]), point_labels=np.array([1]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = timed(f, 10)
        rec("sam2", "ok", model="hiera_large", **r, vram_gb=vram_peak(), measurement_kind="measured")
    except Exception as e:  # noqa: BLE001
        rec("sam2", "failed", err=repr(e)[:300], tb=traceback.format_exc()[-500:])


def gdino():
    pip("transformers accelerate")
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    try:
        mid = "IDEA-Research/grounding-dino-base"
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").half().eval()
        img = Image.open(sample_images()[0]).convert("RGB")
        torch.cuda.reset_peak_memory_stats()

        def f():
            inp = proc(images=img, text="a person. a car. a cat.", return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                model(**inp)
        r = timed(f, 15)
        rec("grounding_dino", "ok", model=mid, impl="transformers", **r, vram_gb=vram_peak(), measurement_kind="measured")
    except Exception as e:  # noqa: BLE001
        rec("grounding_dino", "failed", err=repr(e)[:300], tb=traceback.format_exc()[-500:])


def detectron2():
    import torch
    for cand in (f"/usr/local/cuda-{torch.version.cuda}", f"/usr/local/cuda-{torch.version.cuda.split('.')[0]}"):
        if os.path.isdir(cand):
            os.environ["CUDA_HOME"] = cand
            os.environ["PATH"] = cand + "/bin:" + os.environ["PATH"]
            break
    rc, o, t = pip("'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation", 1500)
    if rc:
        return rec("detectron2", "failed", stage="install", install_s=round(t), log=o)
    import cv2
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda"
    p = DefaultPredictor(cfg)
    img = cv2.imread(sample_images()[0])
    torch.cuda.reset_peak_memory_stats()
    r = timed(lambda: p(img), 20)
    rec("detectron2", "ok", model="mask_rcnn_R_50_FPN_3x", install_s=round(t), **r, vram_gb=vram_peak(), measurement_kind="measured")


def whisper_f5():
    """F5-TTS gera audio com texto conhecido; faster-whisper transcreve (WER contra o texto): mede TTS (RTF) e STT (RTF, WER)."""
    rc, o, t = pip("f5-tts faster-whisper jiwer", 1200)
    pip("-U av", 300)
    pip("nvidia-cublas-cu12 nvidia-cudnn-cu12", 600)  # ctranslate2 (faster-whisper) exige libcublas.so.12 mesmo em imagem CUDA 13
    try:
        import ctypes, glob, site
        for sp in site.getsitepackages():
            for lib in sorted(glob.glob(sp + "/nvidia/cublas/lib/libcublas*.so.12")) + sorted(glob.glob(sp + "/nvidia/cudnn/lib/libcudnn*.so.9")):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:  # noqa: BLE001
        rec("f5_tts", "info", preload_err=repr(e)[:120])
    if rc:
        return rec("f5_tts", "failed", stage="install", log=o)
    texts = {"curto": "Ola, tudo bem? Gostaria de saber o horario de funcionamento da padaria.",
             "medio": "O pedido numero quarenta e dois contem dez paes franceses e dois bolos de cenoura, totalizando sessenta e quatro "
                      "reais. O pagamento pode ser feito por Pix.",
             "longo": " ".join(["A entrega sera feita amanha pela manha no endereco cadastrado e o cliente sera avisado por mensagem."] * 3)}
    try:
        from importlib.resources import files
        from f5_tts.api import F5TTS
        tts = F5TTS()
        ref = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        for k, tx in texts.items():
            t0 = time.time()
            wav, sr, _ = tts.infer(ref_file=ref, ref_text="Some call me nature, others call me mother nature.", gen_text=tx,
                                   file_wave=f"{OUT}/tts_{k}.wav")
            dt = time.time() - t0
            dur = len(wav) / sr
            rec("f5_tts", "ok", case=k, chars=len(tx), audio_s=round(dur, 2), wall_s=round(dt, 2), rtf=round(dt / dur, 3),
                measurement_kind="measured")
    except Exception as e:  # noqa: BLE001
        rec("f5_tts", "failed", err=repr(e)[:300], tb=traceback.format_exc()[-500:])
        return
    try:
        import jiwer
        from faster_whisper import WhisperModel
        wm = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        for k, tx in texts.items():
            t0 = time.time()
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
            aud, sr0 = sf.read(f"{OUT}/tts_{k}.wav", dtype="float32")
            if aud.ndim > 1:
                aud = aud.mean(axis=1)
            aud = resample_poly(aud, 16000, sr0).astype(np.float32)
            segs, info = wm.transcribe(aud, language="pt")
            hyp = " ".join(s.text for s in segs)
            dt = time.time() - t0
            rec("faster_whisper", "ok", case=k, audio_s=round(info.duration, 2), wall_s=round(dt, 2), rtf=round(dt / info.duration, 3),
                wer=round(jiwer.wer(tx.lower(), hyp.lower()), 3), note="ref de voz EN: WER so indicativo", measurement_kind="measured")
    except Exception as e:  # noqa: BLE001
        rec("faster_whisper", "failed", err=repr(e)[:300])


def sdxl():
    pip("diffusers transformers accelerate safetensors", 600)
    import torch
    from diffusers import AutoPipelineForText2Image
    try:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        pr = "foto de uma padaria artesanal ao amanhecer, luz suave, alta qualidade"
        t0 = time.time()
        pipe(pr, num_inference_steps=4, guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(42))
        cold = time.time() - t0
        for bs in (1, 4):
            ts = []
            for i in range(3):
                g = torch.Generator("cuda").manual_seed(42 + i)
                t0 = time.time()
                im = pipe([pr] * bs, num_inference_steps=4, guidance_scale=0.0, generator=g).images
                ts.append(time.time() - t0)
            im[0].save(f"{OUT}/sdxl_bs{bs}.png")
            med = sorted(ts)[1]
            rec("sdxl_turbo", "ok", batch=bs, steps=4, res="512x512", s_per_batch=round(med, 3), s_per_image=round(med / bs, 3),
                first_call_s=round(cold, 2), vram_gb=vram_peak(), measurement_kind="measured")
    except Exception as e:  # noqa: BLE001
        rec("sdxl_turbo", "failed", err=repr(e)[:300], tb=traceback.format_exc()[-500:])


def ltx():
    pip("diffusers transformers accelerate imageio imageio-ffmpeg sentencepiece", 600)
    import torch
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video
    try:
        t0 = time.time()
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
        load = time.time() - t0
        pipe.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        for frames in (25, 49):
            t0 = time.time()
            v = pipe(prompt="padeiro tirando paes frescos do forno, plano medio, luz quente", negative_prompt="borrado, baixa qualidade",
                     width=512, height=320, num_frames=frames, num_inference_steps=25,
                     generator=torch.Generator("cuda").manual_seed(42)).frames[0]
            dt = time.time() - t0
            export_to_video(v, f"{OUT}/ltx_{frames}.mp4", fps=24)
            rec("ltx_video", "ok", frames=frames, res="512x320", steps=25, gen_s=round(dt, 1), video_s=round(frames / 24, 2),
                gen_s_per_video_s=round(dt / (frames / 24), 1), load_s=round(load, 1), vram_gb=vram_peak(), measurement_kind="measured")
    except Exception as e:  # noqa: BLE001
        rec("ltx_video", "failed", err=repr(e)[:300], tb=traceback.format_exc()[-500:])


def ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        sh("apt-get install -y -q ffmpeg", 300)
        ff = shutil.which("ffmpeg")
    if not ff:
        return rec("ffmpeg", "failed", reason="ffmpeg indisponivel")
    enc = sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -E nvenc")[1]
    rec("ffmpeg", "info", nvenc_encoders=enc.strip()[:300])
    for res, nm in (("1920x1080", "1080p"), ("3840x2160", "4k")):
        for codec in ("libx264", "h264_nvenc", "hevc_nvenc"):
            if codec.endswith("nvenc") and "nvenc" not in enc:
                rec("ffmpeg", "incompatible", codec=codec, res=nm, reason="sem nvenc no build")
                continue
            rc, o, t = sh(f"ffmpeg -y -hide_banner -f lavfi -i testsrc2=size={res}:rate=30 -t 20 -c:v {codec} -b:v 6M {OUT}/ff_{nm}_{codec}.mp4", 300)
            rec("ffmpeg", "ok" if rc == 0 else "failed", codec=codec, res=nm, video_s=20, wall_s=round(t, 2),
                speed_x=round(20 / t, 2) if rc == 0 else None, tail=o[-200:] if rc else None, measurement_kind="measured")


def colmap():
    rc, o, t = pip("pycolmap-cuda12", 600)
    if rc:
        pip("pycolmap", 600)
    import pycolmap
    rec("colmap", "info", version=getattr(pycolmap, "__version__", "?"), cuda=getattr(pycolmap, "has_cuda", None))
    d = "/workspace/eco/data/sb"
    os.makedirs(d, exist_ok=True)
    if not glob.glob(d + "/**/*.jpg", recursive=True):
        sh(f"cd {d} && curl -sL -o sb.zip https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip && unzip -q -o sb.zip", 600)
    imgs = sorted(glob.glob(d + "/**/*.jpg", recursive=True))[:40]
    if not imgs:
        return rec("colmap", "skipped", reason="dataset indisponivel")
    sub = d + "/sub"
    os.makedirs(sub, exist_ok=True)
    for f in imgs:
        shutil.copy(f, sub)
    out = d + "/out"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    t0 = time.time()
    pycolmap.extract_features(out + "/db.db", sub)
    t1 = time.time()
    pycolmap.match_exhaustive(out + "/db.db")
    t2 = time.time()
    maps = pycolmap.incremental_mapping(out + "/db.db", sub, out)
    t3 = time.time()
    rec("colmap", "ok", images=len(imgs), extract_s=round(t1 - t0, 1), match_s=round(t2 - t1, 1), mapper_s=round(t3 - t2, 1), models=len(maps),
        summary=str(maps[0].summary())[:300] if maps else None, cuda=getattr(pycolmap, "has_cuda", None), measurement_kind="measured")


def maskrcnn_tv():
    """Proxy de Mask R-CNN (mesma arquitetura do Detectron2) via torchvision; Detectron2 nao compila neste ambiente."""
    import torch
    import torchvision
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).cuda().eval()
    img = torch.rand(3, 800, 1200, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        r = timed(lambda: m([img]), 20)
    rec("maskrcnn_torchvision", "ok", proxy_for="detectron2", model="maskrcnn_resnet50_fpn", torchvision=torchvision.__version__, **r, vram_gb=vram_peak(),
        measurement_kind="measured")


def odm():
    if not os.path.exists("/workspace/eco/data/.extra_done"):
        yolo2()
        track2()
    rec("odm", "incompatible", reason="ODM exige Docker/NodeODM; a instancia Vast nao tem Docker-in-Docker. Medir em VM propria se necessario.")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        t0 = time.time()
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            rec(name, "failed", err=repr(e)[:300], tb=traceback.format_exc()[-600:])
        rec(name, "done", total_s=round(time.time() - t0, 1))

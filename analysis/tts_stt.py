"""F5-TTS (speech synthesis) and faster-whisper (transcription): real-time factor by GPU, cold-start outliers, cost per audio hour, CPU baseline."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import BLUE, GREEN, ORANGE, PURPLE, RED, gpu_name, read, save_fig, save_table, update_summary

libs = read("libs_runs.csv")
libs["gpu"] = libs.gpu.map(gpu_name)
f5 = libs[(libs.lib == "f5_tts") & (libs.status == "ok")].copy()
cold = f5[f5.rtf > 5]
warm = f5[f5.rtf <= 5]
med = warm[warm.case == "medio"].groupby("gpu").agg(rtf=("rtf", "median"), dph=("dph_usd", "median")).sort_values("rtf")
med["usd_per_audio_hour"] = med.rtf * med.dph
save_table(med.round(3), "f5tts_medium_text_warm")
save_table(cold[["gpu", "case", "wall_s", "audio_s", "rtf"]].round(2), "f5tts_cold_start_outliers", index=False)

fw = libs[(libs.lib == "faster_whisper") & (libs.status == "ok")]
save_table(fw[["gpu", "case", "audio_s", "wall_s", "rtf", "wer"]].round(3), "faster_whisper_libs", index=False)
sb = read("sandbox_metrics.csv")
gw = sb[sb.id == "gpu_faster_whisper"].set_index("metric").value
models = {}
for m in ("small", "large_v3_turbo", "large_v3"):
    models[m] = {"rtf": float(gw[f"{m}_rtf"]), "wer": float(gw[f"{m}_wer"])}
wt = pd.DataFrame(models).T
save_table(wt.round(4), "faster_whisper_models_rtx3090")

ov = read("openvoice_cpu_local.csv").iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(med.index, med.rtf, color=BLUE)
axes[0].set_ylabel("Real-time factor (lower is faster), 155-character text")
axes[0].tick_params(axis="x", rotation=40, labelsize=8)
axes[1].bar(med.index, med.usd_per_audio_hour, color=GREEN)
axes[1].set_ylabel("US$ per hour of synthesized audio")
axes[1].tick_params(axis="x", rotation=40, labelsize=8)
axes[2].bar([k.replace("_", "-") for k in wt.index], wt.rtf, color=[ORANGE, PURPLE, BLUE])
axes[2].set_ylabel("faster-whisper real-time factor, RTX 3090")
for i, (k, r) in enumerate(wt.iterrows()):
    axes[2].text(i, r.rtf, f"WER {r.wer:.3f}", ha="center", va="bottom", fontsize=8)
save_fig(fig, "tts_stt.png")

pp = read("media_gated_runs.csv")
pp = pp[pp.tool.str.startswith("NVIDIA PersonaPlex") & (pp.metric == "wall_s")].copy()
pp["gpu"] = pp.gpu.map(gpu_name)
warm_pp = pp[pp.value < 100].groupby("gpu").value.min()
rtf_pp = (warm_pp / 40.0).sort_values()
save_table(rtf_pp.rename("warm_real_time_factor_40s_audio").round(3).to_frame(), "personaplex_warm_rtf")

update_summary({"f5_best_gpu": med.index[0], "f5_best_rtf": med.rtf.iloc[0], "f5_worst_gpu": med.index[-1], "f5_worst_rtf": med.rtf.iloc[-1], "f5_cheapest_usd_per_audio_hour": med.usd_per_audio_hour.min(),
                "f5_cold_max_rtf": cold.rtf.max(), "f5_cold_n": len(cold), "f5_calls": len(f5), "whisper_turbo_rtf": wt.loc["large_v3_turbo", "rtf"], "whisper_large_rtf": wt.loc["large_v3", "rtf"],
                "whisper_turbo_wer": wt.loc["large_v3_turbo", "wer"], "openvoice_convert_rtf_cpu": ov.convert_rtf, "openvoice_melo_rtf_cpu": ov.melo_rtf,
                "personaplex_best_rtf": rtf_pp.iloc[0], "personaplex_worst_rtf": rtf_pp.iloc[-1], "personaplex_gpus": len(rtf_pp)})
print(med.round(3)); print(wt.round(3)); print(rtf_pp.round(3))

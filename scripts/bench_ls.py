#!/usr/bin/env python3
"""LatentSync 測速：模型載入一次，連續處理一批片段；記錄載入時間、每段耗時、GPU 功耗、顯存峰值。

cd ~/LatentSync && .venv/bin/python bench_ls.py 輸入資料夾 輸出資料夾 結果.json
輸入資料夾裡是 xxx_video.mp4 + xxx_audio.wav 成對的檔案。參數和 inference.sh 一樣：stage2_512、30 步、guidance 1.5、DeepCache。
"""
import glob, json, os, platform, subprocess, sys, threading, time
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from accelerate.utils import set_seed
from DeepCache import DeepCacheSDHelper

src, dst, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
STEPS, GUIDE = 30, 1.5


class Power(threading.Thread):
    """每 0.5 秒讀一次 nvidia-smi 功耗（W）和顯存（MiB，GB10 統一記憶體會是 N/A）。"""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.on = [], True

    def run(self):
        while self.on:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(",")
            try:
                self.samples.append((time.time(), float(r[0]), r[1].strip()))
            except ValueError:
                pass
            time.sleep(0.5)

    def between(self, t0, t1):
        w = [p for t, p, _ in self.samples if t0 <= t <= t1]
        return (sum(w) / len(w) if w else None), (max(w) if w else None)


pw = Power(); pw.start()
config = OmegaConf.load("configs/unet/stage2_512.yaml")
dtype = torch.float16
t_load0 = time.time()
scheduler = DDIMScheduler.from_pretrained("configs")
audio_encoder = Audio2Feature(model_path="checkpoints/whisper/tiny.pt", device="cuda",
                              num_frames=config.data.num_frames, audio_feat_length=config.data.audio_feat_length)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
vae.config.scaling_factor, vae.config.shift_factor = 0.18215, 0
unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model), "checkpoints/latentsync_unet.pt", device="cpu")
pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype), scheduler=scheduler).to("cuda")
helper = DeepCacheSDHelper(pipe=pipe); helper.set_params(cache_interval=3, cache_branch_id=0); helper.enable()
torch.cuda.synchronize(); t_load = time.time() - t_load0

res = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "steps": STEPS, "guidance": GUIDE, "load_s": round(t_load, 1), "clips": []}
for v in sorted(glob.glob(f"{src}/*_video.mp4")):
    n = os.path.basename(v)[:-len("_video.mp4")]
    a = f"{src}/{n}_audio.wav"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a],
                               capture_output=True, text=True).stdout)
    set_seed(1247); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    pipe(video_path=v, audio_path=a, video_out_path=f"{dst}/{n}_ls.mp4", num_frames=config.data.num_frames,
         num_inference_steps=STEPS, guidance_scale=GUIDE, weight_dtype=dtype, width=config.data.resolution,
         height=config.data.resolution, mask_image_path=config.data.mask_image_path, temp_dir=f"{dst}/tmp_{n}")
    torch.cuda.synchronize(); t1 = time.time()
    avg_w, max_w = pw.between(t0, t1)
    c = {"clip": n, "audio_s": round(dur, 2), "time_s": round(t1 - t0, 1), "x_realtime": round((t1 - t0) / dur, 1),
         "peak_torch_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
         "avg_w": round(avg_w, 1) if avg_w else None, "max_w": round(max_w, 1) if max_w else None,
         "energy_wh": round(avg_w * (t1 - t0) / 3600, 2) if avg_w else None}
    res["clips"].append(c); print(c, flush=True)
pw.on = False
tot = sum(c["time_s"] for c in res["clips"]); aud = sum(c["audio_s"] for c in res["clips"])
res["total_s"], res["total_audio_s"] = round(tot, 1), round(aud, 1)
res["total_wh"] = round(sum(c["energy_wh"] or 0 for c in res["clips"]), 1)
json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
print("TOTAL", res["load_s"], "s load,", tot, "s for", aud, "s audio,", res["total_wh"], "Wh")

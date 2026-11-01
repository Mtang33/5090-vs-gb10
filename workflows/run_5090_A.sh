#!/bin/bash
# 實戰對比 A 組（5090）：關鍵幀 → Wan 鏡頭 → H3 鏡頭，各自用該用的 ComfyUI
cd ~/clay; PY=~/ComfyUI/.venv/bin/python
start_comfy() {  # $1=port $2=extra args
  pkill -f "[m]ain.py --listen 127.0.0.1 --port" ; sleep 5
  (cd ~/ComfyUI && setsid nohup .venv/bin/python main.py --listen 127.0.0.1 --port $1 $2 --disable-auto-launch > ~/comfyui_$1.log 2>&1 < /dev/null &)
  for i in $(seq 1 90); do curl -s 127.0.0.1:$1/system_stats >/dev/null && return 0; sleep 2; done; echo "ComfyUI $1 起不來"; exit 1
}
echo "$(date +%T) keys"; start_comfy 8189 ""
$PY bench_comfy.py --machine 5090 keys clay_scenes.json || echo KEYS_FAIL
echo "$(date +%T) anim"; start_comfy 8188 "--use-sage-attention"
$PY bench_comfy.py --machine 5090 anim clay_shots.json --keys ~/clay/keys || echo ANIM_FAIL
echo "$(date +%T) h3"
$PY bench_comfy.py --machine 5090 h3 flf_shots.json --keys ~/clay/flf_keys || echo H3_FAIL
pkill -f "[m]ain.py --listen 127.0.0.1 --port"
echo "$(date +%T) ALL_DONE"

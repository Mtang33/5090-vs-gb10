#!/usr/bin/env bash
# CASCADE 集群耗电对齐：DeepSeek（生产 B4-2u）在线时跑一段固定负载，记下每段起止时间。
# 插座功率由 Mac 上的 tapo_power.py 同时记录（.29 = 4 台 DGX + 交换机），第二天用 marks.csv 对齐。
# 阶段：待机 5 分 → E02 四题 ×3（频道自己的活）→ L3 解码/prefill（quick）→ 待机 3 分
set -euo pipefail
OUT=${PRODQ_DIR:-$HOME/fabric-ab/results/power_ds_$(date +%m%d_%H%M)}
mkdir -p "$OUT"
M="$OUT/marks.csv"
echo "phase,start_epoch,end_epoch" > "$M"
mark() { local p=$1; shift; local t0; t0=$(date +%s.%N); "$@"; echo "$p,$t0,$(date +%s.%N)" >> "$M"; }
cd ~/fabric-ab
curl -fsS -m 5 http://127.0.0.1:8888/v1/models | grep -q deepseek || { echo "DeepSeek 不在线"; exit 2; }
mark idle_pre sleep 300
for i in 1 2 3; do
  mark "e02_eval_$i" python3 eval4/run_eval.py "power_ds_$i" --url http://127.0.0.1:8888 --model deepseek-v4.1-flash --ctk '{"thinking": false}'
  cp "eval4/results/power_ds_$i.json" "$OUT/" 2>/dev/null || true
done
mark l3_quick python3 l3_serving.py "$OUT/l3_quick.json" --quick --only decode,prefill
mark idle_post sleep 180
echo "完成：$M"; cat "$M"

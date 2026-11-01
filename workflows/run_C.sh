#!/bin/bash
# 實戰對比 C 組（4 台 GB10 並行，在 dgx-head 上跑）。分工照真實做法：每台拿一部分場景/鏡頭，接力鏡頭（s3→s3b、s7→s7b→s7c、s8→s8b）放同一台。
# 每台自己寫 CSV（machine=GB10x4）；整體耗時 = 最晚結束 − 最早開始。
NODES=(localhost spark-msi spark-worker spark-3)
KEYS=("s1 s2 s3" "s4 s5" "s6 s7" "s8 s9")
ANIM=("s7 s7b s7c s1" "s3 s3b s2" "s8 s8b s4" "s5 s6 s9")
EDIT=("e1 e2" "e3 e4" "e5" "e6")
H3R=("r1" "r2" "r3 r4" "r5 r6")
on() { if [[ $1 == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes $1 "$2"; fi; }
PY='cd ~/clay_bench && ~/ComfyUI/.venv/bin/python bench_comfy.py --machine GB10x4'
ts() { date +%s.%N | cut -c1-14; }
echo "$(date +%T) idle-baseline"; sleep 60

echo "C_STAGE comfy start $(ts)"
for i in 0 1 2 3; do
  cmd="$PY keys clay_scenes.json ${KEYS[$i]} && $PY anim clay_shots.json ${ANIM[$i]} --keys ~/clay_bench/keys"
  [[ $i == 1 ]] && cmd="$cmd && $PY h3 flf_shots.json --keys ~/clay_bench/flf_keys"
  on ${NODES[$i]} "$cmd" > ~/clay_bench/C_comfy_$i.log 2>&1 &
done; wait
echo "C_STAGE comfy end $(ts)"; sleep 30

echo "C_STAGE edit+h3ref start $(ts)"
for i in 0 1 2 3; do
  on ${NODES[$i]} "$PY edit qedit_jobs.json ${EDIT[$i]} --keys ~/clay_bench/keys && $PY h3 h3ref_shots.json ${H3R[$i]} --keys ~/clay_bench/keys" > ~/clay_bench/C_edit_$i.log 2>&1 &
done; wait
echo "C_STAGE edit+h3ref end $(ts)"; sleep 30

# Remotion：短片 1807 幀、長片 0-5999 幀各均分 4 段（無聲，參數同 remotion_bench.sh），最後在 head 上拼接
render() {  # $1=comp $2=總幀起 $3=總幀止 $4=並發 $5=標籤
  local a=$2 b=$3 n=$(( $3 - $2 + 1 )) i s e
  for i in 0 1 2 3; do
    s=$(( a + n * i / 4 )); e=$(( a + n * (i + 1) / 4 - 1 ))
    on ${NODES[$i]} "cd ~/video_studio && mkdir -p out/bench && PATH=~/.local/node/bin:\$PATH npx remotion render src/index.ts $1 out/bench/$5_p$i.mp4 --frames=$s-$e --muted --concurrency=$4 --crf=18 --offthreadvideo-threads=4 > out/bench/$5_p$i.log 2>&1" &
  done; wait
  for i in 1 2 3; do scp -q ${NODES[$i]}:video_studio/out/bench/$5_p$i.mp4 ~/video_studio/out/bench/; done
  ( cd ~/video_studio/out/bench && printf "file '%s'\n" $5_p0.mp4 $5_p1.mp4 $5_p2.mp4 $5_p3.mp4 > $5.txt && ffmpeg -v error -y -f concat -safe 0 -i $5.txt -c copy $5.mp4 )
}
echo "C_STAGE remotion-short start $(ts)"; render Bashi02Clay 0 1806 12 short_x4; echo "C_STAGE remotion-short end $(ts)"; sleep 20
echo "C_STAGE remotion-long start $(ts)"; render BashiLongFull 0 5999 8 long_x4; echo "C_STAGE remotion-long end $(ts)"
echo "$(date +%T) ALL_DONE"

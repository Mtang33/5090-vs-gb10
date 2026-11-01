#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp（llama-server）跑分：跟 bench_5090_gb10.py 用同一份提示、同样 512 token、3 次取中位数，结果格式也一样，
可以直接用 bench_power_merge.py 对插座功率。

用法（pc5090，llama-server 已在 127.0.0.1:8080 跑着某个模型）：
  python3 bench_llamacpp.py --machine 5090 --label gpt-oss:120b@ncmoe22
  python3 bench_llamacpp.py --machine 5090 --label gpt-oss:120b@ncmoe22 --long
结果：同目录 bench_<machine>_llamacpp-<short|long>_<日期时间>.csv
"""
import argparse, csv, json, os, statistics, sys, time, urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_5090_gb10 import prompt, long_prompt, GpuSampler   # 同一份口播稿和提示


def complete(host, p, n, ctx):
    body = {"prompt": p, "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": False, "stream": False}
    req = urllib.request.Request(host + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode())
    t = d["timings"]
    return dict(prompt_tokens=t["prompt_n"], prefill_tok_s=round(t["prompt_per_second"], 1),
                gen_tokens=t["predicted_n"], gen_tok_s=round(t["predicted_per_second"], 1),
                first_token_s=round(t["prompt_ms"] / 1000, 2), start_epoch=round(t0, 1), end_epoch=round(time.time(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--label", required=True, help="写进 model 栏，例如 gpt-oss:120b@ncmoe22")
    ap.add_argument("--host", default="http://127.0.0.1:8080")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args()
    mode = "llamacpp-long" if a.long else "llamacpp-short"
    props = json.load(urllib.request.urlopen(a.host + "/props"))
    ctx = props.get("default_generation_settings", {}).get("n_ctx")
    print(f"机器 {a.machine} · llama.cpp {props.get('build_info', '')} · 上下文 {ctx} · {a.label}")
    complete(a.host, prompt("warm"), 16, ctx)                          # 预热
    rows = []
    for i in range(a.runs):
        g = GpuSampler(); g.start()
        r = complete(a.host, long_prompt(i) if a.long else prompt(i), 128 if a.long else a.gen, ctx)
        g.stop_flag = True; g.join(2)
        r.update(machine=a.machine, mode=mode, model=a.label, run=i + 1,
                 gpu_power_avg_w=round(statistics.mean(g.power)) if g.power else "",
                 gpu_power_peak_w=round(max(g.power)) if g.power else "", time=datetime.now().isoformat(timespec="seconds"))
        rows.append(r)
        print(f"  第 {i + 1} 次  读提示 {r['prefill_tok_s']} tok/s（{r['prompt_tokens']} token） · "
              f"生成 {r['gen_tok_s']} tok/s（{r['gen_tokens']} token） · 首字 {r['first_token_s']} 秒")
    med = lambda k: statistics.median(x[k] for x in rows)
    print(f"中位数：读提示 {med('prefill_tok_s')} · 生成 {med('gen_tok_s')} tok/s")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"bench_{a.machine}_{mode}_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("结果已写入：", path)


if __name__ == "__main__":
    main()

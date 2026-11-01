#!/usr/bin/env python3
"""把 tapo_power.py 記的插座功率對到跑分結果上：每次測試的整機平均 / 峰值功率、耗電、每千 token 耗電。

python3 工具/bench_power_merge.py power_*.csv bench_*.csv [--idle 300]
→ 每個 bench csv 旁邊寫一個 *_插座功率.csv，並印出按「機器 × 測試 × 模型」的中位數表。
--idle：每台機器取記錄裡最安靜的連續 N 秒當待機功率（預設 300 秒）。

共用插頭（例如 4 台 DGX + 交換機插在同一個）量不到單台整機功率，所以另給「增量」：
測試時的平均功率減掉該插頭的待機功率 = 正在跑的那台因為負載多用的電；兩台都用增量比，才公平。
"""
import argparse, csv, statistics, sys
from collections import defaultdict


def load_power(paths):
    pw = defaultdict(list)                       # 機器 → [(epoch, W)]
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["watts"]:
                pw[r["machine"]].append((float(r["epoch"]), float(r["watts"])))
    for k in pw:
        pw[k].sort()
    return pw


def window(series, t0, t1):
    return [w for t, w in series if t0 <= t <= t1]


def idle(series, secs):
    best = None
    for i, (t, _) in enumerate(series):
        win = [w for tt, w in series[i:] if tt <= t + secs]
        if len(win) >= secs * 0.8:
            m = statistics.mean(win)
            best = m if best is None or m < best else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--idle", type=int, default=300)
    a = ap.parse_args()
    pfiles = [f for f in a.files if "power_" in f.split("/")[-1]]
    bfiles = [f for f in a.files if "bench_" in f.split("/")[-1]]
    if not pfiles or not bfiles:
        sys.exit("要同時給 power_*.csv 和 bench_*.csv")
    pw = load_power(pfiles)
    base = {}
    for m, s in pw.items():
        base[m] = idle(s, a.idle)
        print(f"{m}：記錄 {len(s)} 筆，待機（最安靜 {a.idle} 秒）約 {base[m]:.1f} W" if base[m] else f"{m}：記錄 {len(s)} 筆")
    table = defaultdict(list)
    for bf in bfiles:
        rows = list(csv.DictReader(open(bf, encoding="utf-8-sig")))
        out = []
        for r in rows:
            series = pw.get(r["machine"], [])
            t0, t1 = float(r["start_epoch"]), float(r["end_epoch"])
            win = window(series, t0, t1)
            if not win:
                r.update(wall_avg_w="", wall_peak_w="", wall_wh="", wh_per_1k_tok="", delta_w="", delta_wh_per_1k_tok="")
            else:
                avg = statistics.mean(win)
                wh = avg * (t1 - t0) / 3600
                toks = float(r.get("gen_tokens") or 0)
                if r.get("mode", "").startswith("parallel") and r.get("parallel_total_tok_s"):
                    toks = float(r["parallel_total_tok_s"]) * (t1 - t0)
                b = base.get(r["machine"])
                dw = avg - b if b else None
                dwh = dw * (t1 - t0) / 3600 if dw is not None else None
                r.update(wall_avg_w=round(avg, 1), wall_peak_w=round(max(win), 1), wall_wh=round(wh, 3),
                         wh_per_1k_tok=round(wh / toks * 1000, 3) if toks else "",
                         delta_w=round(dw, 1) if dw is not None else "",
                         delta_wh_per_1k_tok=round(dwh / toks * 1000, 3) if toks and dwh is not None else "")
                table[(r["machine"], r.get("mode", ""), r["model"])].append(
                    (avg, max(win), r["wh_per_1k_tok"], r["delta_w"], r["delta_wh_per_1k_tok"]))
            out.append(r)
        dst = bf.replace(".csv", "_插座功率.csv")
        with open(dst, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print("→", dst)
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    print(f"\n{'機器':6}{'測試':12}{'模型':38}{'插座平均W':>10}{'峰值W':>8}{'增量W':>8}{'Wh/千token':>12}{'增量Wh/千token':>16}")
    for (m, mode, model), v in sorted(table.items()):
        print(f"{m:6}{mode:12}{model:38}{med([x[0] for x in v]):>10.1f}{max(x[1] for x in v):>8.1f}"
              f"{med([x[3] for x in v if x[3] != '']):>8.1f}{med([x[2] for x in v if x[2] != '']):>12.3f}"
              f"{med([x[4] for x in v if x[4] != '']):>16.3f}")


if __name__ == "__main__":
    main()

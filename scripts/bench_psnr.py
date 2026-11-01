#!/usr/bin/env python3
"""實戰對比畫質：同一份工作流、同 seed，比兩台機器的輸出（bench_comfy.py 存的 ~/bench_comfy/<machine>/<task>/<id>/NNNN.png）。
每個任務/鏡頭算全部幀的平均 PSNR 和最低一幀；圖片類只有一幀。≥40 dB 肉眼基本看不出，30–40 細節有差，<30 構圖可能都不一樣了。

  python3 bench_psnr.py ~/qc/5090 ~/qc/GB10 [~/qc/GB10x4 ...]   # 第一個是基準，後面每個都跟它比
輸出 CSV：bench_psnr_<日期時間>.csv（本腳本旁邊）
"""
import csv, glob, os, sys
from datetime import datetime
import numpy as np
from PIL import Image


def psnr(a, b):
    if a.shape != b.shape: return None
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def frames(d): return sorted(f for f in glob.glob(f'{d}/[0-9]*.png'))


def main():
    base, others = sys.argv[1], sys.argv[2:]
    out = []
    for task in ('keys', 'edit', 'anim', 'h3'):
        for d in sorted(glob.glob(f'{base}/{task}/*/')):
            i = os.path.basename(d.rstrip('/'))
            if i.startswith('_'): continue
            fa = frames(d)
            for o in others:
                fb = frames(f'{o}/{task}/{i}')
                if not fb or len(fb) != len(fa):
                    print(f'{task}/{i} vs {os.path.basename(o)}: 幀數不同或缺 ({len(fa)} / {len(fb)})'); continue
                vals = [psnr(np.asarray(Image.open(x).convert('RGB')), np.asarray(Image.open(y).convert('RGB'))) for x, y in zip(fa, fb)]
                vals = [v for v in vals if v is not None]
                fin = [v for v in vals if v != float('inf')]
                avg = sum(fin) / len(fin) if fin else float('inf')
                row = dict(task=task, id=i, base=os.path.basename(base), other=os.path.basename(o), frames=len(vals),
                           psnr_avg=round(avg, 2), psnr_min=round(min(vals), 2), identical=sum(v == float('inf') for v in vals))
                out.append(row)
                print(f"{task:5s} {i:5s} {row['base']}↔{row['other']:7s} {len(vals):4d} 幀  平均 {row['psnr_avg']:6.2f} dB  最低 {row['psnr_min']:6.2f}  完全相同 {row['identical']} 幀", flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'bench_psnr_{datetime.now():%Y%m%d_%H%M}.csv')
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print('結果：', p)


if __name__ == '__main__':
    main()

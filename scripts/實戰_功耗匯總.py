#!/usr/bin/env python3
"""實戰對比 A/B/C 三組：每項任務的耗時和插座增量電量（Wh）。
增量 = 區間內插座讀數積分 − 該插座閒置基線 × 時長；基線取每組開跑前的閒置窗口中位數。
A 組量 .56（只接 5090），B/C 組量 .29（4 台 DGX + 交換機）。"""
import csv, glob, os, statistics
from datetime import datetime

D = os.path.dirname(os.path.abspath(__file__))
E = lambda s: datetime.fromisoformat(s).timestamp()      # 本機時區（和插座記錄、日誌一致）


def load_power(path):
    pts = {}
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        pts.setdefault(r['machine'], []).append((float(r['epoch']), float(r['watts'])))
    return pts


P_A = load_power(f'{D}/實戰A組_5090/power_20261101_0122.csv')
P_1 = load_power(f'{D}/實戰B組_GB10/power_20261101_0523.csv')
P_2 = load_power(glob.glob(f'{D}/實戰C組_GB10x4/power_20261101_09*.csv')[0])


def base(pts, a, b): return statistics.median(w for t, w in pts if a <= t <= b)


def wh(pts, a, b, bl):
    """梯形積分 [a,b] 的增量 Wh；回傳 (增量 Wh, 平均增量 W, 覆蓋率)"""
    seg = [(t, w) for t, w in pts if a <= t <= b]
    if len(seg) < 2: return None, None, 0
    e = sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(seg, seg[1:]))
    span = seg[-1][0] - seg[0][0]
    return (e - bl * span) / 3600 * (b - a) / span, e / span - bl, span / (b - a)


def rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if '插座' in p: continue                          # bench_power_merge 的合併版，行是重複的
        out += list(csv.DictReader(open(p, encoding='utf-8-sig')))
    return out


def task(pts, bl, rs):
    """一組 CSV 行：總秒數、每行中位秒數、增量 Wh 合計"""
    secs = [float(r['seconds']) for r in rs]
    e = [wh(pts, float(r['start_epoch']), float(r['end_epoch']), bl)[0] for r in rs]
    return sum(secs), statistics.median(secs), sum(x for x in e if x is not None), len(rs)


BL = {
    'A_clay': base(P_A['5090'], E('2026-11-01T01:29:00'), E('2026-11-01T01:31:50')),
    'A_more': base(P_1['5090'], 1793532455.7, 1793532485.7),
    'B': base(P_1['DGX'], E('2026-11-01T05:24:15'), E('2026-11-01T05:25:12')),
    'B_rem': base(P_2['DGX'], 1793547319.8, 1793547349.8),
    'C': base(P_2['DGX'], E('2026-11-01T09:39:51'), E('2026-11-01T09:40:48')),
}
print('閒置基線 W：', {k: round(v, 1) for k, v in BL.items()})

A, B, C = f'{D}/實戰A組_5090', f'{D}/實戰B組_GB10', f'{D}/實戰C組_GB10x4'
print('\n## 逐項（不含預熱）：總秒數 / 每個中位秒數 / 增量 Wh / 個數')
for name, pa, pb in [('① 關鍵幀', 'keys_20261101_0133', 'keys'), ('② Wan 鏡頭', 'anim_20261101_0147', 'anim'),
                     ('H3 首尾幀', 'h3_20261101_0148', 'h3_20261101_0619'), ('⑦ H3 參考圖', 'h3_20261101_0542', 'h3_20261101_0639'),
                     ('⑧ 修圖', 'edit', 'edit')]:
    ra = rows(f'{A}/bench_5090_comfy-{pa}*.csv')
    pts_a, bl_a = (P_A['5090'], BL['A_clay']) if '0133' in pa or '0147' in pa or '0148' in pa else (P_1['5090'], BL['A_more'])
    rb = rows(f'{B}/bench_GB10_comfy-{pb}*.csv')
    ta, tb = task(pts_a, bl_a, ra), task(P_1['DGX'], BL['B'], rb)
    print(f'{name:10s} A {ta[0]:6.0f}s {ta[1]:5.1f}s {ta[2]:6.1f}Wh n={ta[3]} | B {tb[0]:6.0f}s {tb[1]:5.1f}s {tb[2]:6.1f}Wh n={tb[3]}')

print('\n## 整段（含載入/預熱，牆鐘）：秒 / 增量 Wh / 平均增量 W')
def seg(label, pts, bl, a, b):
    e, w, cov = wh(pts, a, b, bl)
    print(f'{label:28s} {b - a:7.0f}s  {e:6.1f} Wh  +{w:5.0f} W  覆蓋 {cov:.0%}')
seg('A 粘土 short（keys+Wan+H3）', P_A['5090'], BL['A_clay'], E('2026-11-01T01:32:02'), E('2026-11-01T01:48:47'))
seg('B 粘土 short（keys+Wan+H3）', P_1['DGX'], BL['B'], E('2026-11-01T05:25:13'), E('2026-11-01T06:19:53'))
seg('C 粘土 short（4 台並行）', P_2['DGX'], BL['C'], 1793547649.080, 1793548650.053)
seg('B 修圖+H3參考圖', P_1['DGX'], BL['B'], E('2026-11-01T06:20:32'), E('2026-11-01T06:39:58'))
seg('C 修圖+H3參考圖', P_2['DGX'], BL['C'], 1793548680.058, 1793549119.972)
seg('A Remotion 短片', P_1['5090'], BL['A_more'], 1793532485.7, 1793532522.1)
seg('A Remotion 長片 6000', P_1['5090'], BL['A_more'], 1793532542.1, 1793532676.9)
seg('B Remotion 短片', P_2['DGX'], BL['B_rem'], 1793547349.8, 1793547391.1)
seg('B Remotion 長片 6000', P_2['DGX'], BL['B_rem'], 1793547411.1, 1793547559.0)
for line in open(f'{C}/run_BC.log', encoding='utf-8'):
    if line.startswith('C_STAGE remotion'):
        k = line.split(); globals().setdefault('_st', {})[(k[1], k[2])] = float(k[3])
for kind in ('short', 'long'):
    seg(f'C Remotion {kind}', P_2['DGX'], BL['C'], _st[(f'remotion-{kind}', 'start')], _st[(f'remotion-{kind}', 'end')])

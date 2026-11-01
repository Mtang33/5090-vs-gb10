#!/usr/bin/env python3
"""FunASR 轉寫跑分（11/4 那期 A4）：和 ep_asr.py 同一套模型（paraformer-zh + fsmn-vad + ct-punc），同一段 E02 錄音。
載入時間單獨記；先預熱一次（不計），再正式轉 3 次取中位數。CSV 格式同 bench_5090_gb10.py，可用 bench_power_merge.py 合併插座功率。

  python bench_asr.py --machine 5090 E02T_B.wav [--runs 3]
"""
import sys, setuptools._distutils as _d; sys.modules["distutils"] = _d   # Python 3.12 沒有 distutils
import argparse, csv, os, statistics, time, wave
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True)
    ap.add_argument('wav')
    ap.add_argument('--runs', type=int, default=3)
    a = ap.parse_args()
    with wave.open(a.wav) as w: dur = w.getnframes() / w.getframerate()
    import torch
    from funasr import AutoModel
    t0 = time.time()
    m = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True)
    print(f'{a.machine} 載入模型 {time.time() - t0:.1f}s，裝置 {m.kwargs.get("device")}，音頻 {dur:.1f}s', flush=True)
    t0 = time.time(); m.generate(input=a.wav, sentence_timestamp=True, batch_size_s=300); print(f'  預熱 {time.time() - t0:.1f}s', flush=True)
    rows = []
    for k in range(1, a.runs + 1):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time(); r = m.generate(input=a.wav, sentence_timestamp=True, batch_size_s=300)[0]; t1 = time.time()
        rows.append((k, t0, t1, len(r['text'])))
        print(f'  第 {k} 次 {t1 - t0:.2f}s · 實時倍數 {dur / (t1 - t0):.0f}× · {len(r["text"])} 字', flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'bench_{a.machine}_asr_{datetime.now():%Y%m%d_%H%M}.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['machine', 'mode', 'model', 'run', 'audio_seconds', 'seconds', 'rtf_x', 'chars', 'start_epoch', 'end_epoch', 'time'])
        for k, t0, t1, n in rows:
            w.writerow([a.machine, 'asr', 'paraformer-zh+fsmn-vad+ct-punc', k, round(dur, 1), round(t1 - t0, 2), round(dur / (t1 - t0)),
                        n, round(t0, 1), round(t1, 1), datetime.fromtimestamp(t0).isoformat(timespec='seconds')])
    med = statistics.median(t1 - t0 for _, t0, t1, _ in rows)
    print(f'中位數 {med:.2f}s（{dur / med:.0f} 倍實時）\n結果： {path}')


if __name__ == '__main__':
    main()

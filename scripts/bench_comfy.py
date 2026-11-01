#!/usr/bin/env python3
"""實戰對比：用《巴士談》粘土 short 的同一份場景/鏡頭表，在本機 ComfyUI 上重跑並計時。
工作流直接 import clay_gen.py / clay_anim.py / h3_anim.py（和 10-24 出片時同一份），只換 URL。
每個任務記開始/結束時間（和 tapo_power.py 對齊）、顯卡功耗和顯存峰值；結果格式同 bench_5090_gb10.py，可直接用 bench_power_merge.py 合併插座功率。

在跑 ComfyUI 的那台機器上執行（腳本和 clay_*.py、h3_anim.py 放同一目錄）：
  python3 bench_comfy.py --machine 5090 keys ~/clay/clay_scenes.json                 # ① Qwen-Image 關鍵幀，每場景 1 張
  python3 bench_comfy.py --machine 5090 anim ~/clay/clay_shots.json --keys ~/clay/keys       # ② Wan 2.2 i2v 動畫鏡頭
  python3 bench_comfy.py --machine 5090 h3   ~/clay/flf_shots.json  --keys ~/clay/flf_keys   # ⑤⑦ H3（首尾幀或參考圖）
  python3 bench_comfy.py --machine 5090 h3   ~/clay/h3ref_shots.json --keys ~/clay/keys       # ⑦ H3 參考圖生視頻（鏡頭給 refs）
  python3 bench_comfy.py --machine 5090 edit ~/clay/qedit_jobs.json --keys ~/clay/keys       # ⑧ Qwen-Image-Edit 2511 修圖
  加 id 只跑那幾個：... anim ~/clay/clay_shots.json s1 s2；C 組（4 台分攤）就是每台給不同的 id。
輸出：~/bench_comfy/<machine>/<task>/<id>/（留著跟另一台比 PSNR）；CSV 在本腳本旁邊。
"""
import argparse, csv, os, shutil, statistics, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clay_gen, clay_anim, h3_anim, qedit                # noqa: E402
from bench_5090_gb10 import GpuSampler                    # noqa: E402

COMFY = os.path.expanduser('~/ComfyUI')


def save_frames(imgs, out):
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    for k, im in enumerate(imgs):
        src = f"{COMFY}/output/{im['subfolder']}/{im['filename']}"
        shutil.move(src, f'{out}/{k:04d}.png')
    shutil.copy(f'{out}/{len(imgs) - 1:04d}.png', f'{out}/last.png')
    return len(imgs)


def timed(fn):
    g = GpuSampler(); g.start(); t0 = time.time()
    res = fn()
    t1 = time.time(); g.stop_flag = True; g.join(3)
    return res, t0, t1, (round(statistics.mean(g.power)) if g.power else ''), (round(max(g.power)) if g.power else ''), \
        (round(max(g.mem) / 1024, 1) if g.mem else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True)
    ap.add_argument('task', choices=['keys', 'anim', 'h3', 'edit'])
    ap.add_argument('config')
    ap.add_argument('ids', nargs='*')
    ap.add_argument('--keys', default=os.path.expanduser('~/clay/keys'), help='關鍵幀所在目錄（鏡頭表裡的 key 檔名相對於它）')
    ap.add_argument('--qwen-url', default='http://127.0.0.1:8189')
    ap.add_argument('--wan-url', default='http://127.0.0.1:8188')
    ap.add_argument('--no-warmup', action='store_true')
    a = ap.parse_args()
    import json
    cfg = json.load(open(os.path.expanduser(a.config)))
    out_root = os.path.expanduser(f'~/bench_comfy/{a.machine}/{a.task}')
    os.makedirs(out_root, exist_ok=True)
    clay_gen.URL = a.qwen_url; clay_anim.URL = h3_anim.URL = a.wan_url
    rows = []

    if a.task == 'keys':
        items = [s for s in cfg['scenes'] if not a.ids or s['id'] in a.ids]
        def job(sc, steps=4, warm=False):
            # 預熱換 seed 和提示詞：ComfyUI 會快取一模一樣的任務（0 秒返回），計時就不準了
            pos = f"{sc['prompt']} {cfg['style']}" + (' (warmup)' if warm else '')
            seed = sc.get('seed', 1000) + (99991 if warm else 0)
            img = clay_gen.run(clay_gen.wf(pos, sc.get('negative') or cfg['negative'], seed, cfg['w'], cfg['h'], steps))
            return save_frames([img], f"{out_root}/{'_warmup' if warm else sc['id']}")
        if not a.no_warmup:
            print('預熱（載入模型，不計）…', flush=True); _, t0, t1, *_ = timed(lambda: job(items[0], warm=True)); print(f'  載入+首張 {t1 - t0:.1f}s')
        for sc in items:
            (n, t0, t1, pw, pk, mem) = timed(lambda: job(sc))
            rows.append(dict(id=sc['id'], frames=n, t0=t0, t1=t1, pw=pw, pk=pk, mem=mem))
            print(f"  {sc['id']}: {t1 - t0:.1f}s · 顯卡 {pw} W · 顯存峰值 {mem} GB", flush=True)

    elif a.task == 'edit':
        items = [e for e in cfg['edits'] if not a.ids or e['id'] in a.ids]
        def job(e, warm=False):
            name = f"bench_{a.machine}_edit_{e['id']}.png"; shutil.copy(os.path.join(os.path.expanduser(a.keys), e['key']), f'{COMFY}/input/{name}')
            img = qedit.run(qedit.wf(name, e['prompt'] + (' (warmup)' if warm else ''), e.get('seed', 1) + (99991 if warm else 0)))
            return save_frames([img], f"{out_root}/{'_warmup' if warm else e['id']}")
        if not a.no_warmup:
            print('預熱（載入模型，不計）…', flush=True); _, t0, t1, *_ = timed(lambda: job(items[0], warm=True)); print(f'  載入+首張 {t1 - t0:.1f}s')
        for e in items:
            (n, t0, t1, pw, pk, mem) = timed(lambda: job(e))
            rows.append(dict(id=e['id'], frames=n, t0=t0, t1=t1, pw=pw, pk=pk, mem=mem))
            print(f"  {e['id']}: {t1 - t0:.1f}s · 顯卡 {pw} W · 顯存峰值 {mem} GB", flush=True)

    else:
        shots = cfg['shots'] if not a.ids else [s for s in cfg['shots'] if s['id'] in a.ids]
        def src_of(s):
            if s.get('from_last'):
                p = f"{out_root}/{s['from_last']}/last.png"
                return p if os.path.exists(p) else None
            return os.path.join(os.path.expanduser(a.keys), s['key'])
        def job(s, length=None, warm=False):
            src = src_of(s) if not s.get('refs') else os.path.join(os.path.expanduser(a.keys), s['refs'][0])
            if src is None:
                raise RuntimeError(f"{s['id']} 要接 {s['from_last']} 的最後一幀，先跑那個鏡頭")
            name = f"bench_{a.machine}_{a.task}_{s['id']}.png"; shutil.copy(src, f'{COMFY}/input/{name}')
            pos = f"{s['motion']} {cfg['style']}" + (' (warmup)' if warm else '')   # 預熱換提示詞和 seed，避開 ComfyUI 快取
            seed = s.get('seed', 42) + (99991 if warm else 0)
            L = length or s.get('length', cfg.get('length', 81))
            if a.task == 'anim':
                imgs = clay_anim.run(clay_anim.wf(name, pos, cfg['negative'], cfg['w'], cfg['h'], L, seed))
            elif s.get('refs'):   # ⑦ 參考圖生視頻（ref2va），refs 相對於 --keys
                refs = []
                for k, r in enumerate(s['refs']):
                    refs.append(f"bench_{a.machine}_h3_{s['id']}_ref{k}.png"); shutil.copy(os.path.join(os.path.expanduser(a.keys), r), f'{COMFY}/input/{refs[-1]}')
                imgs = h3_anim.run(h3_anim.wf_r2v(refs, pos, cfg['w'], cfg['h'], L, seed, s.get('steps', 4)))
            else:
                last = None
                if s.get('last'):
                    last = f"bench_{a.machine}_h3_{s['id']}_last.png"; shutil.copy(os.path.join(os.path.expanduser(a.keys), s['last']), f'{COMFY}/input/{last}')
                imgs = h3_anim.run(h3_anim.wf(name, last, pos, cfg['w'], cfg['h'], L, seed, s.get('steps', 6)))
            return save_frames(imgs, f"{out_root}/{'_warmup' if warm else s['id']}")
        if not a.no_warmup:
            w = next(s for s in shots if not s.get('from_last'))
            print('預熱（載入模型 + 17 幀短片，不計）…', flush=True)
            _, t0, t1, *_ = timed(lambda: job(w, length=17, warm=True)); print(f'  載入+預熱 {t1 - t0:.1f}s')
        for s in shots:
            (n, t0, t1, pw, pk, mem) = timed(lambda: job(s))
            rows.append(dict(id=s['id'], frames=n, t0=t0, t1=t1, pw=pw, pk=pk, mem=mem))
            print(f"  {s['id']}: {n} 幀 {t1 - t0:.0f}s · 顯卡 {pw} W · 顯存峰值 {mem} GB", flush=True)

    path = os.path.join(HERE, f"bench_{a.machine}_comfy-{a.task}_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['machine', 'mode', 'model', 'run', 'frames', 'seconds', 'gpu_power_avg_w', 'gpu_power_peak_w', 'gpu_mem_peak_gb',
                    'start_epoch', 'end_epoch', 'time'])
        for k, r in enumerate(rows, 1):
            w.writerow([a.machine, f'comfy-{a.task}', f"{a.task}:{r['id']}", k, r['frames'], round(r['t1'] - r['t0'], 1), r['pw'], r['pk'],
                        r['mem'], round(r['t0'], 1), round(r['t1'], 1), datetime.fromtimestamp(r['t0']).isoformat(timespec='seconds')])
    tot = sum(r['t1'] - r['t0'] for r in rows)
    print(f'共 {len(rows)} 個，總計 {tot:.0f} 秒（{tot / 60:.1f} 分）· 中位數 {statistics.median(r["t1"] - r["t0"] for r in rows):.1f} 秒/個')
    print('結果：', path)


if __name__ == '__main__':
    main()

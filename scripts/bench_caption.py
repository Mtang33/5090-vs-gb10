#!/usr/bin/env python3
"""實戰對比 ⑨ 字幕校對：把一份 srt 切成 N 塊，逐塊交給本地 Ollama 模型校對（改錯字、標點），計時。
企劃原本用巴士談逐字稿，但那是敏感內容；這裡改用 E02 自己的繁中字幕，兩台同一份輸入、同一個模型（qwen3:30b-a3b-instruct）。
只比耗時和耗電，不比校對品質。CSV 格式同 bench_5090_gb10.py，可用 bench_power_merge.py 合併插座功率。

  python3 bench_caption.py --machine 5090 E02_成片v4_字幕_繁中.srt [--blocks 12] [--url http://127.0.0.1:11434]
輸出：CSV 在本腳本旁邊；校對結果在 ~/bench_caption/<machine>/block_NN.txt
"""
import argparse, csv, json, os, re, time, urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = 'qwen3:30b-a3b-instruct-2507-q4_K_M'
PROMPT = ('下面是一段繁體中文口播影片的字幕，每行一句，行首是編號。請校對：改錯別字、同音字和明顯的標點問題，'
          '不要改寫句子、不要合併或拆分行。逐行輸出「編號<TAB>校對後的句子」，不要輸出其他內容。\n\n')


def read_srt(p):
    blocks = re.split(r'\n\s*\n', open(p, encoding='utf-8-sig').read().strip())
    lines = []
    for b in blocks:
        rows = b.strip().splitlines()
        if len(rows) >= 3:
            lines.append((rows[0].strip(), ' '.join(rows[2:]).strip()))
    return lines


def chat(url, text):
    body = {'model': MODEL, 'stream': False, 'messages': [{'role': 'user', 'content': PROMPT + text}],
            'options': {'temperature': 0, 'seed': 1, 'num_ctx': 8192}}
    r = urllib.request.Request(url + '/api/chat', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True)
    ap.add_argument('srt')
    ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--url', default='http://127.0.0.1:11434')
    a = ap.parse_args()
    lines = read_srt(a.srt)
    n = a.blocks; size = -(-len(lines) // n)
    chunks = ['\n'.join(f'{i}\t{t}' for i, t in lines[k:k + size]) for k in range(0, len(lines), size)]
    out = os.path.expanduser(f'~/bench_caption/{a.machine}'); os.makedirs(out, exist_ok=True)
    print(f'{len(lines)} 句，切成 {len(chunks)} 塊；預熱（載入模型，不計）…', flush=True)
    t0 = time.time(); chat(a.url, chunks[0][:200]); print(f'  載入+預熱 {time.time() - t0:.1f}s', flush=True)
    rows = []
    for k, c in enumerate(chunks, 1):
        t0 = time.time(); r = chat(a.url, c); t1 = time.time()
        open(f'{out}/block_{k:02d}.txt', 'w', encoding='utf-8').write(r['message']['content'])
        pe, ev = r.get('prompt_eval_count', 0), r.get('eval_count', 0)
        gen = ev / (r['eval_duration'] / 1e9) if r.get('eval_duration') else 0
        rows.append(dict(k=k, t0=t0, t1=t1, pe=pe, ev=ev, gen=gen))
        print(f'  塊 {k:2d}: {t1 - t0:5.1f}s · 輸入 {pe} · 輸出 {ev} token · 生成 {gen:.1f} tok/s', flush=True)
    path = os.path.join(HERE, f'bench_{a.machine}_caption_{datetime.now():%Y%m%d_%H%M}.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['machine', 'mode', 'model', 'run', 'prompt_tokens', 'gen_tokens', 'seconds', 'gen_tok_s', 'start_epoch', 'end_epoch', 'time'])
        for r in rows:
            w.writerow([a.machine, 'caption', MODEL, r['k'], r['pe'], r['ev'], round(r['t1'] - r['t0'], 1), round(r['gen'], 1),
                        round(r['t0'], 1), round(r['t1'], 1), datetime.fromtimestamp(r['t0']).isoformat(timespec='seconds')])
    tot = sum(r['t1'] - r['t0'] for r in rows)
    print(f'共 {len(rows)} 塊，總計 {tot:.0f} 秒（{tot / 60:.1f} 分）· 輸出 {sum(r["ev"] for r in rows)} token')
    print('結果：', path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""L3 服務級：解碼、短提示 TTFT、長輸入 prefill（needle）、批量長輸出。在 dgx-head 上跑，直連 127.0.0.1:8888。
每項先熱身一次（丟棄），再正式跑 REPEATS 次。每項開始前等服務空閒。輸入都是合成的，不用任何播客素材。

用法：python3 l3_serving.py <輸出.json> [--quick] [--repeats N] [--only decode,ttft,prefill,batch]
"""
import concurrent.futures as cf, json, random, statistics, sys, time, urllib.request
from common import DS_URL, ds_load

import os
MODEL = os.environ.get("L3_MODEL", "deepseek-v4.1-flash")
CTK = json.loads(os.environ.get("L3_CTK", '{"thinking": false}'))   # GLM：L3_CTK='{"reasoning_effort":"low"}'
REASONING = bool(os.environ.get("L3_REASONING"))   # 關不掉思考的模型：思考 token 也算進解碼；prefill 改用首字時間
WORDS_PER_TOKEN = 0.366  # 2026-10-27 用隨機英文詞校準：約 2.73 token/詞
# 詞表用 Mac 的 /usr/share/dict/words（部署時拷成同目錄的 words.txt），校準就是用它做的
_WF = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "words.txt")
WORDS = [w.strip() for w in open(_WF) if w.strip().isalpha()]
PROMPTS = {
    "prose": "用中文写一篇约一千字的散文，主题是秋天傍晚在老城区散步时看到的街景、人和声音。不要列提纲，直接写正文。",
    "code": "Write a complete Python module implementing an LRU cache with a doubly linked list and dictionary. "
            "Include get, put, delete, iteration, resize, clear, invariant validation and detailed docstrings. Return code only.",
}
TOPICS = ["城市里的夜班公交", "第一次学游泳", "老电影院拆除那天", "雨季的菜市场", "一台旧收音机", "外婆的腌菜坛",
          "地铁末班车", "冬天的澡堂", "小区里的流浪猫", "高考结束那个下午", "搬家时翻到的信", "深夜便利店",
          "山里的小学", "第一次出国", "旧书店的老板", "春节的火车站"]


def post(body, stream=False, timeout=1800):
    req = urllib.request.Request(DS_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_one(content, max_tokens, temperature=0.0):
    """返回 (ttft_s, decode_tok_s, completion_tokens, prompt_tokens, wall_s)。"""
    body = {"model": MODEL, "temperature": temperature, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": CTK,
            "messages": [{"role": "user", "content": content}]}
    t0 = time.perf_counter(); first = last = None; usage = {}
    with post(body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                if dl.get("content") or (REASONING and (dl.get("reasoning_content") or dl.get("reasoning"))):
                    now = time.perf_counter(); first = first or now; last = now
    ct = usage.get("completion_tokens", 0)
    dec = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else None
    return (first - t0 if first else None), dec, ct, usage.get("prompt_tokens"), time.perf_counter() - t0


def wait_idle(max_s=600):
    t0 = time.time()
    while time.time() - t0 < max_s:
        if ds_load() == (0, 0):
            return
        time.sleep(3)
    raise RuntimeError("等了 10 分鐘服務還不空閒")


def words(n_tokens, rnd):
    return " ".join(rnd.choice(WORDS) for _ in range(int(n_tokens * WORDS_PER_TOKEN)))


def med(xs):
    xs = [x for x in xs if x is not None]
    return {"median": round(statistics.median(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2),
            "n": len(xs)} if xs else None


# ---------------------------------------------------------------- 各項

def t_decode(cfg, rnd):
    out = {}
    for kind in cfg["kinds"]:
        for c in cfg["levels"]:
            runs = []
            for rep in range(cfg["repeats"] + 1):  # 第 0 次是熱身
                wait_idle()
                tags = [f"{kind}-{c}-{rep}-{i}-{rnd.random():.6f}" for i in range(c)]
                t0 = time.perf_counter()
                with cf.ThreadPoolExecutor(c) as ex:
                    rs = list(ex.map(lambda t: stream_one(f"[{t}] {PROMPTS[kind]}", cfg["dec_tokens"]), tags))
                wall = time.perf_counter() - t0
                agg = sum(r[2] for r in rs) / wall
                per = statistics.median([r[1] for r in rs if r[1]])
                if rep:
                    runs.append({"agg_tok_s": round(agg, 1), "per_stream_tok_s": round(per, 1),
                                 "ttft_s": round(statistics.median([r[0] for r in rs if r[0]]), 3)})
            out[f"{kind}_c{c}"] = {"runs": runs, "agg": med([r["agg_tok_s"] for r in runs]),
                                   "per_stream": med([r["per_stream_tok_s"] for r in runs])}
            print(f"  解碼 {kind:5} C{c:<2}  合計 {out[f'{kind}_c{c}']['agg']['median']:>6} tok/s  "
                  f"每路 {out[f'{kind}_c{c}']['per_stream']['median']:>5} tok/s", flush=True)
    return out


def t_ttft(cfg, rnd):
    out = {}
    for n in cfg["ttft_tokens"]:
        xs = []
        for rep in range(cfg["ttft_n"] + 1):
            wait_idle()
            r = stream_one(f"[{rnd.random()}] " + words(n, rnd) + "\n\n用一个词概括上面的内容。", 4)
            if rep:
                xs.append(r[0])
        out[f"{n // 1000}k"] = med(xs)
        print(f"  TTFT {n // 1000:>3}k  中位 {out[f'{n // 1000}k']['median']} s", flush=True)
    return out


def t_prefill(cfg, rnd):
    out = {}
    for n in cfg["prefill_tokens"]:
        runs = []
        for rep in range(cfg["repeats"] + (0 if n > 200_000 else 1)):  # 259k 不熱身，太久
            wait_idle()
            code = str(rnd.randint(100000, 999999))
            body = words(n, rnd).split(" ")
            body.insert(len(body) // 2, f". IMPORTANT: the secret code is {code}. ")
            prompt = ("Below is a long list of random words with a secret code inside.\n\n" + " ".join(body)
                      + "\n\nWhat is the secret code? Answer with the number only.")
            t0 = time.perf_counter()
            if REASONING:   # 思考模型：先想再答，16 token 不夠；prefill 速度用「提示 token ÷ 首字時間」，needle 在思考和答案裡都找
                body = {"model": MODEL, "temperature": 0, "max_tokens": 400, "stream": True, "chat_template_kwargs": CTK,
                        "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
                first = None; ans = think = ""; usage = {}   # 答案和思考分開收：推測解碼一塊裡可能同時有思考結尾和答案開頭，混在一起會把數字拆開
                with post(body, stream=True) as r:
                    for raw in r:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        ev = json.loads(line[5:]); usage = ev.get("usage") or usage
                        for ch in ev.get("choices") or []:
                            dl = ch.get("delta") or {}
                            a_, t_ = dl.get("content") or "", dl.get("reasoning_content") or dl.get("reasoning") or ""
                            if a_ or t_:
                                first = first or time.perf_counter(); ans += a_; think += t_
                wall = (first or time.perf_counter()) - t0
                pt = usage["prompt_tokens"]; ok = code in ans or code in think; text = f"[答案] {ans} [思考] {think}"
            else:
                with post({"model": MODEL, "temperature": 0, "max_tokens": 16, "chat_template_kwargs": CTK,
                           "messages": [{"role": "user", "content": prompt}]}) as r:
                    d = json.load(r)
                wall = time.perf_counter() - t0
                pt = d["usage"]["prompt_tokens"]
                ok = code in d["choices"][0]["message"]["content"]
            if rep or n > 200_000:
                runs.append({"prompt_tokens": pt, "wall_s": round(wall, 1), "tok_s": round(pt / wall), "needle_ok": ok,
                             **({} if ok else {"code": code, "got": (text if REASONING else d["choices"][0]["message"]["content"])[-300:]})})
        out[f"{n // 1000}k"] = {"runs": runs, "tok_s": med([r["tok_s"] for r in runs]),
                                "needle_all_ok": all(r["needle_ok"] for r in runs)}
        print(f"  prefill {n // 1000:>3}k  {out[f'{n // 1000}k']['tok_s']['median']} tok/s  "
              f"needle {'全對' if out[f'{n // 1000}k']['needle_all_ok'] else '✗ 有錯'}", flush=True)
    return out


def t_batch(cfg, rnd):
    runs = []
    for rep in range(cfg["repeats"] + 1):
        wait_idle()
        n, c = cfg["batch_n"], cfg["batch_c"]
        reqs = [f"写一篇约 500 字的短文，主题：{TOPICS[i % len(TOPICS)]}。（编号 {rep}-{i}-{rnd.random():.5f}）"
                for i in range(n)]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(c) as ex:
            rs = list(ex.map(lambda p: stream_one(p, cfg["batch_tokens"], 0.3), reqs))
        wall = time.perf_counter() - t0
        if rep:
            runs.append({"wall_s": round(wall, 1), "tok_s": round(sum(r[2] for r in rs) / wall, 1),
                         "avg_tokens": round(statistics.mean(r[2] for r in rs))})
    out = {"runs": runs, "tok_s": med([r["tok_s"] for r in runs])}
    print(f"  批量 {cfg['batch_n']} 個 C{cfg['batch_c']}  {out['tok_s']['median']} tok/s", flush=True)
    return out


def main():
    a = sys.argv[1:]
    out_path = next((x for x in a if not x.startswith("--") and not x.isdigit() and "," not in x), None)
    quick = "--quick" in a
    reps = int(a[a.index("--repeats") + 1]) if "--repeats" in a else (1 if quick else 3)
    only = a[a.index("--only") + 1].split(",") if "--only" in a else ["decode", "ttft", "prefill", "batch"]
    cfg = dict(repeats=reps, kinds=["prose", "code"], levels=[1, 4, 8, 16], dec_tokens=512,
               ttft_tokens=[1000, 8000], ttft_n=5, prefill_tokens=[32_000, 131_000, 259_000],
               batch_n=32, batch_c=16, batch_tokens=700)
    if quick:
        cfg.update(kinds=["prose"], levels=[1, 4], dec_tokens=128, ttft_tokens=[1000], ttft_n=2,
                   prefill_tokens=[8_000], batch_n=4, batch_c=4, batch_tokens=128)
    if "--levels" in a:   # 只測部分並發，例如 --levels 1（監控輪詢開銷這類只看單路的對比）
        cfg["levels"] = [int(x) for x in a[a.index("--levels") + 1].split(",")]
    if os.environ.get("L3_PREFILL"):   # GLM 的分詞器把同樣的字切得更多，259k 會超過 262144 上下文，用 250k
        cfg["prefill_tokens"] = [int(x) for x in os.environ["L3_PREFILL"].split(",")]
    if ds_load() is None:
        sys.exit("✗ DeepSeek 不在跑")
    rnd = random.Random(time.time())
    res = {"config": cfg, "started": time.strftime("%F %T")}
    fns = {"decode": t_decode, "ttft": t_ttft, "prefill": t_prefill, "batch": t_batch}
    for k in only:
        print(f"== {k}", flush=True)
        t0 = time.time()
        res[k] = fns[k](cfg, rnd)
        res[k + "_wall_s"] = round(time.time() - t0)
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)   # 每項跑完就存，中途失敗也不丟前面的結果
    res["finished"] = time.strftime("%F %T")
    if out_path:
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
        print(f"已寫入 {out_path}")


if __name__ == "__main__":
    main()

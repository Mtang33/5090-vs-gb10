#!/usr/bin/env python3
"""第 4 階段：同一份輸入、同一份提示跑一個模型，存原始輸出 + 自動評分。

  python3 run_eval.py <標籤> --url http://127.0.0.1:8888 --model deepseek-v4.1-flash [--ctk '{"thinking": false}']

結果寫 results/<標籤>.json。溫度 0；思考能關就關（各模型的關法不同，用 --ctk 傳）。
"""
import argparse
import difflib
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INP = json.load(open(os.path.join(HERE, "inputs.json"), encoding="utf-8"))

P_T1 = """下面是一段口播視頻錄製時的語音轉寫，每行一句，格式「編號 [開始秒-結束秒] 文字」。錄製時有說錯重來、口誤、開錄前後的閒話和多餘的語氣詞。
你是剪輯師，要剪成一條乾淨的成片：說錯重來的只留最後說對的那遍，刪掉口誤、重複、場外話；正常內容保留。
只輸出一行 JSON：{{"delete": [要刪掉的句子編號]}}，不要解釋。

{lines}"""
P_T2 = """下面是口播視頻的語音轉寫字幕，每行一條，有同音錯字、漏字、斷句問題。請逐行校對：只改錯字和明顯的轉寫錯誤，不改說話人的用詞和語氣，不合併也不拆分行。
輸出和輸入同樣行數，每行一條校對後的字幕，不要編號，不要解釋。

{lines}"""
P_T3 = """下面是一條約 4 分鐘的 YouTube 口播視頻的全部文字，主題是播客剪輯師到底剪掉了什麼。
請給出：
1. 4–6 個章節：每個章節一行，格式「章節名｜這一章開頭的那句話（原文照抄前 15 個字）」
2. 3 個視頻標題候選，每個不超過 25 個字，適合中文 YouTube。

{text}"""
P_T4 = """閱讀下面四份文件，回答文末的問題。每題一行，格式「題號. 答案」，答案要簡短並帶上原文裡的數字。

{corpus}

問題：
{qs}"""


def chat(url, model, ctk, prompt, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if ctk:
        body["chat_template_kwargs"] = ctk
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    msg = d["choices"][0]["message"]
    text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
    u = d.get("usage", {})
    return {"text": text, "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            "wall_s": round(wall, 1), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "finish": d["choices"][0].get("finish_reason")}


def norm(s):
    return re.sub(r"[\s\W_]+", "", s)


def score_t1(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        dele = set(int(x) for x in json.loads(m.group(0))["delete"])
    except Exception:
        return {"parse_error": True}
    ids = [s["id"] for s in INP["t1"]["sentences"]]
    truth_keep = set(INP["t1"]["truth_keep"])
    truth_del = set(ids) - truth_keep
    tp = len(dele & truth_del)
    acc = sum((i in dele) == (i in truth_del) for i in ids) / len(ids)
    prec = tp / len(dele) if dele else 0.0
    rec = tp / len(truth_del) if truth_del else 0.0
    return {"accuracy": round(acc, 3), "delete_precision": round(prec, 3), "delete_recall": round(rec, 3),
            "delete_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "n_delete": len(dele), "n_truth_delete": len(truth_del),
            "keep_all_baseline": round(len(truth_keep) / len(ids), 3)}


def score_t2(text):
    script = norm("".join(INP["t2"]["script"]))
    asr = norm("".join(INP["t2"]["asr"]))
    out_lines = [l for l in text.splitlines() if l.strip()]
    out = norm("".join(out_lines))
    sim = lambda a: difflib.SequenceMatcher(None, a, script, autojunk=False).ratio()
    return {"asr_vs_script": round(sim(asr), 4), "out_vs_script": round(sim(out), 4),
            "gain": round(sim(out) - sim(asr), 4), "lines_in": len(INP["t2"]["asr"]), "lines_out": len(out_lines)}


def score_t4(text):
    qa = INP["t4"]["qa"]
    got = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[\.、．:：)]\s*(.*)", line)
        if m:
            got[int(m.group(1))] = m.group(2)
    per = [any(k in got.get(i + 1, "") for k in q["keys"]) for i, q in enumerate(qa)]
    return {"correct": sum(per), "of": len(qa), "per_question": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctk", default="")
    ap.add_argument("--max-tokens", type=int, default=6000)
    a = ap.parse_args()
    ctk = json.loads(a.ctk) if a.ctk else None
    t1 = INP["t1"]["sentences"]
    jobs = {
        "t1": P_T1.format(lines="\n".join(f"{s['id']} [{s['start']}-{s['end']}] {s['text']}" for s in t1)),
        "t2": P_T2.format(lines="\n".join(INP["t2"]["asr"])),
        "t3": P_T3.format(text=INP["t3"]["text"]),
        "t4": P_T4.format(corpus=INP["t4"]["corpus"],
                          qs="\n".join(f"{i + 1}. {q['q']}" for i, q in enumerate(INP["t4"]["qa"]))),
    }
    res = {"label": a.label, "model": a.model, "ctk": ctk, "started": time.strftime("%F %T")}
    for k, prompt in jobs.items():
        r = chat(a.url, a.model, ctk, prompt, a.max_tokens)
        r["tok_s"] = round(r["completion_tokens"] / r["wall_s"], 1) if r.get("completion_tokens") else None
        if k == "t1":
            r["score"] = score_t1(r["text"])
        elif k == "t2":
            r["score"] = score_t2(r["text"])
        elif k == "t4":
            r["score"] = score_t4(r["text"])
        res[k] = r
        print(f"  {k}: {r['wall_s']} s, {r['completion_tokens']} tok, finish={r['finish']}, "
              f"score={r.get('score', '（盲評）')}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"{a.label}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wrote results/{a.label}.json")


if __name__ == "__main__":
    main()

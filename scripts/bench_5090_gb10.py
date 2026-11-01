#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX 5090 vs GB10 跑分脚本（湯包實驗室 11/4 那期）
只用 Python 标准库，Linux / Windows 通用。需要本机已装好 Ollama 并在运行，两台 Ollama 版本一致。

用法（10-31 修订：5090 = pc5090 Ubuntu，GB10 = dgx-worker）：
  python3 bench_5090_gb10.py --machine 5090                 # 基本测试：短提示（约 1,100 token），六个模型
  python3 bench_5090_gb10.py --machine 5090 --long          # 长提示（约 3 万 token，像整期逐字稿），上下文 40960
  python3 bench_5090_gb10.py --machine 5090 --parallel 4    # 同时 4 个请求的总吞吐（服务端要 OLLAMA_NUM_PARALLEL>=4）
  python3 bench_5090_gb10.py --machine GB10 --soak 180 --soak-model gpt-oss:20b   # 持续满载 3 分钟，读功率计用
  python3 bench_5090_gb10.py --machine GB10 --soak 3600 --soak-model gpt-oss:120b # 长时间满载，看会不会降频

结果：同目录下 bench_<machine>_<日期时间>.csv，每次正式测一行；跑完打印中位数汇总。
"""
import argparse, csv, json, os, platform, random, shutil, statistics, subprocess, sys, threading, time, urllib.request, urllib.error
from datetime import datetime

MODELS = [
    ("qwen3:8b",                            "小模型，两边都装得下，看纯速度"),
    ("gpt-oss:20b",                         "中型 MoE，很多人在家跑的那种"),
    ("qwen3:30b-a3b-instruct-2507-q4_K_M",  "GB10 上平时在用的逐句标删字模型（频道主线）"),
    ("qwen3.6:35b",                         "2026 年的新 MoE"),
    ("qwen3.8:27b",                         "2026 年的新稠密模型，接近 5090 显存上限"),
    ("gpt-oss:120b",                        "约 65GB，5090 显存装不下要分到内存；GB10 整个装得下"),
]
LONG_MODELS = ["gpt-oss:20b", "qwen3:30b-a3b-instruct-2507-q4_K_M", "gpt-oss:120b"]   # 长提示只测这三个，太费时

TEXT = """露脸。先念剪前示范句，再放剪后版本；下方波形同步，被剪的部分红色高亮。你听，这两句话，意思一模一样，第二句短了一半。上一期我留了个问题：我剪一期播客，通常只剪掉 19%，可平均每分钟要下 10 刀。一期三个小时的录音，剪完还剩两个多小时。那这 19% 到底剪的是啥？今天咱们就把它拆开看看。数据图①：剪掉的时间 64% 停顿空白 / 30% 整段 / 6% 逐字精剪；再切数据图②：131 刀的长短分布。我把能用的几十期工程都算了一遍。被剪掉的时间里，六成多其实是停顿和空白；三成是整段拿掉的，一拿就是十几秒往上，比如开录前的试音，或者聊跑偏的一大段。但你要是数刀数，那就完全反过来了。我拿一段访谈的开头数过：七分钟的录音，我下了一百三十一刀，三秒多就一刀。这一百三十一刀里，有八十多刀剪掉的东西不到零点八秒。就是一个字、两个字，一个“然后”，一个“就是”。所以按时间算，大头是空白和整段；按功夫算，大头是这些小刀。我剪一期花三四个小时，基本都花在这儿了。每种一张标题卡，接示范句剪前／剪后和波形。每种 30–40 秒。我把这些小刀归了一下类，大概有七种。你可以听听，看看自己平时说话中了几种。第一种，最常见的：句子开头的“然后”“所以说”“就是说”“我觉得”。不过注意啊，不是见到“然后”就剪。人家要是真的在讲先后顺序，那就得留着。我剪的是那种纯粹嘴在热身的“然后”。第二种，话说了一半，推翻，重新说。人说话很少一次就说对。现场听着没问题，录下来就会觉得磕磕绊绊。我只留改过的那一遍。这种得挑，哪遍说得好留哪遍。很多时候是第二遍才说顺。④⑥ 有示范句；⑦ 用两轨示意动画（谁说话开谁的麦）。这种一刀下去就是两三秒，甚至五六秒，拿掉之后主线清楚多了。难就难在，有的岔路特别好玩，那就不剪。这个分寸，说实话我自己都讲不太清楚。第五种，场外的东西：开录前试音，中间调麦，“等一下，我重新说”，还有猫跳上桌子。这是唯一一种我会整段整段删的。刚才说的那三成整段拿掉的，大部分就是它。第六种，说到一半卡住，在想词。半秒到两秒的停顿，连着拖长的音，我会压到几乎没有。但句子跟句子之间，我会留一口气。全压掉的话，听的人会喘不过气。第七种，很多人想不到：别人说话的时候，你那条轨上的“嗯”“对”“哈哈”。这个我不剪，我是把那条轨关掉。每个人一条麦嘛，谁在说话就开谁的，其他人的都关上。所以成品里，几乎任何时候都只有一条麦是开着的。我算过，我手剪的部分，两个人的麦同时开着的时间，不到百分之零点五。这就是为什么剪过的对谈听起来特别干净：没有串音，底噪也不会叠在一起。咳嗽、清嗓子也一样，旁边有人在说话，我就只关掉咳嗽那条轨；没人说话，才整段剪掉。说完剪什么，其实更重要的，是不剪什么。第一，嘉宾说的实质内容，我基本不动。我剪的是“怎么说”，很少剪“说什么”。第二，带语气的词我会留着。比如“其实”“当然”，听着像口头禅，可很多时候它是在表态度。第三，剪完的句子一定得是顺的。宁可少剪一个字，也不能留半句话。还有一种最微妙的：对方正说到兴头上，你在旁边“哈哈”笑了一声。有时候它就是噪音，有时候它就是那段对话的节奏。留不留，我靠的是感觉。而这个，恰恰是我最想看看 AI 能不能学会的。数据图③：每期剪掉比例，标出中位数 19%、最狠 42%。当然，19% 只是一个中位数。大部分期在一成到两成多之间，每分钟八到十四刀。也有剪得特别狠的：有一期我剪掉了四成多。还有一期更夸张，开头两个小时整段没用，因为正题到第三个小时才开始。所以，没有一个“该剪多少”的标准答案。我每一期其实都在判断：这期的主线是什么，哪些话在帮它，哪些话在拖它。七种刀总览图卡，切到代码编辑器。看到这儿你可能会想：这不就是七条规则嘛，写个程序不就完事了？我一开始也是这么想的。然后我真写了，翻车翻得挺惨。下一期，我把这些翻车现场都摆出来给你看。觉得有意思的话，订阅一下；剪辑上有什么问题，也留在评论区。"""

def prompt(i):
    # 每次开头放不同编号，避免 Ollama 复用上一次的提示缓存，保证每次都真的算一遍 prefill
    nonce = random.randint(100000, 999999)
    return (f"测试编号 {nonce}-{i}。下面是一段播客剪辑视频的口播稿。\n\n{TEXT}\n\n"
            "请先用五条要点总结这段口播稿，然后逐条展开解释，每条至少写三句话。")

def long_prompt(i, parts=24):
    """约 3 万 token：口播稿重复 24 段，每段编号不同，模拟整期逐字稿；要求只做很短的回答，主要测读提示速度。"""
    nonce = random.randint(100000, 999999)
    body = "\n\n".join(f"【第 {k + 1} 段 · {nonce}】{TEXT}" for k in range(parts))
    return (f"测试编号 {nonce}-{i}。下面是一整期播客的逐字稿（共 {parts} 段）。\n\n{body}\n\n"
            "请用三句话说明这期在讲什么。")


def api(host, path, body=None, timeout=3600):
    url = host.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

class GpuSampler(threading.Thread):
    """每秒读一次 nvidia-smi 的显卡功耗和显存（GB10 上功耗可能读不到，读不到就留空）"""
    def __init__(self):
        super().__init__(daemon=True)
        self.ok = shutil.which("nvidia-smi") is not None
        self.power, self.mem, self.stop_flag = [], [], False
    def run(self):
        while self.ok and not self.stop_flag:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,memory.used",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                p, m = [x.strip() for x in out.split(",")]
                if p not in ("[N/A]", "N/A", ""): self.power.append(float(p))
                if m not in ("[N/A]", "N/A", ""): self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(1)
    def stop(self):
        self.stop_flag = True
        return (round(statistics.mean(self.power)) if self.power else "",
                round(max(self.power)) if self.power else "")

def sysinfo():
    info = {"os": platform.platform(), "cpu": platform.machine()}
    if shutil.which("nvidia-smi"):
        try:
            info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
    return info

def mem_check():
    # Linux（GB10）上提醒：DeepSeek / ComfyUI 没停的话，统一内存不够，测出来的数不准
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    gb = int(line.split()[1]) / 1024 / 1024
                    print(f"可用内存 {gb:.0f} GB")
                    if gb < 100:
                        print("⚠️  可用内存不到 100 GB：先停掉这台上的 DeepSeek / ComfyUI 再测，否则 120b 装不下、其他数也偏低。")
                        if sys.stdin.isatty() and input("仍然继续？(y/N) ").strip().lower() != "y":   # 后台跑（无终端）只警告不停
                            sys.exit(1)
    except FileNotFoundError:
        pass

def gen(host, model, p, n, ctx=8192):
    return api(host, "/api/generate", {
        "model": model, "prompt": p, "stream": False, "keep_alive": "15m",
        "options": {"temperature": 0, "seed": 42, "num_ctx": ctx, "num_predict": n,
                    **({"num_batch": int(os.environ["BENCH_NUM_BATCH"])} if os.environ.get("BENCH_NUM_BATCH") else {})}})
    # BENCH_NUM_BATCH：qwen3.6:35b 在 GB10 上 num_batch 512/1024 会 CUDA 崩溃，补测时两台都设 2048


def parallel_round(host, model, n_req, n_tok, i):
    """同时发 n_req 个请求，回传（总生成 token / 墙钟秒数，单个请求生成速度的中位数）。"""
    out = [None] * n_req
    def one(k):
        out[k] = gen(host, model, prompt(f"{i}-{k}"), n_tok)
    t0 = time.time()
    th = [threading.Thread(target=one, args=(k,)) for k in range(n_req)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    toks = sum(r.get("eval_count", 0) for r in out)
    per = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in out if r.get("eval_duration")]
    return round(toks / wall, 1), round(statistics.median(per), 1), toks, round(wall, 1)

def placement(host, model):
    try:
        for m in api(host, "/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                return round(size / 1e9, 1), (round(100 * vram / size) if size else "")
    except Exception:
        pass
    return "", ""

def unload(host, model):
    try:
        api(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
    except Exception:
        pass

def soak(host, model, seconds, machine):
    print(f"\n持续满载 {seconds} 秒（{model}）。插座功率由 tapo_power.py 自动记录。")
    gen(host, model, prompt(0), 16)
    t0 = time.time(); end, i, s = t0 + seconds, 0, GpuSampler(); s.start()
    rates = []
    while time.time() < end:
        i += 1
        r = gen(host, model, prompt(i), 512)
        rates.append(r['eval_count'] / (r['eval_duration'] / 1e9))
        print(f"  第 {i} 轮  生成 {rates[-1]:.1f} tok/s   还剩 {max(int(end - time.time()), 0)} 秒")
    avg, peak = s.stop()
    t1 = time.time()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bench_{machine}_soak_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["machine", "mode", "model", "rounds", "gen_tok_s_first", "gen_tok_s_last", "gen_tok_s_median",
                    "gpu_power_avg_w", "gpu_power_peak_w", "start_epoch", "end_epoch"])
        w.writerow([machine, "soak", model, i, round(rates[0], 1), round(rates[-1], 1), round(statistics.median(rates), 1),
                    avg, peak, round(t0, 1), round(t1, 1)])
    if avg != "":
        print(f"显卡功耗（nvidia-smi）：平均 {avg} W，峰值 {peak} W")
    print(f"生成速度：开头 {rates[0]:.1f} → 结尾 {rates[-1]:.1f} tok/s（看有没有降频）· 结果 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, help="5090 或 GB10（写进结果文件名）")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=512, help="每次生成多少 token")
    ap.add_argument("--models", nargs="*", help="只测这几个；默认测全部六个（--long 只测三个）")
    ap.add_argument("--soak", type=int, default=0, help="持续满载秒数（读功率计、录噪音用）")
    ap.add_argument("--soak-model", default="gpt-oss:20b")
    ap.add_argument("--long", action="store_true", help="长提示测试（约 3 万 token，上下文 40960）")
    ap.add_argument("--parallel", type=int, default=0, help="同时 N 个请求测总吞吐")
    a = ap.parse_args()
    mode = "long" if a.long else f"parallel{a.parallel}" if a.parallel else "short"
    ctx = 40960 if a.long else 8192 * max(1, a.parallel)

    try:
        ver = api(a.host, "/api/version")["version"]
    except Exception as e:
        sys.exit(f"连不上 Ollama（{a.host}）：{e}\n先打开 Ollama 再跑。")
    info = sysinfo()
    print(f"机器 {a.machine} · Ollama {ver} · {info}")
    mem_check()

    if a.soak:
        soak(a.host, a.soak_model, a.soak, a.machine); return

    have = {m["name"] for m in api(a.host, "/api/tags").get("models", [])}
    todo = [m for m, _ in MODELS if (not a.models or m in a.models)]
    if a.long and not a.models:
        todo = [m for m in todo if m in LONG_MODELS]
    if a.parallel:
        print(f"并发测试：服务端要设 OLLAMA_NUM_PARALLEL>={a.parallel}（否则请求会排队，测出来等于单个）")
    missing = [m for m in todo if m not in have]
    if missing:
        print("这些模型还没下载，会跳过：\n  " + "\n  ".join(f"ollama pull {m}" for m in missing))
    todo = [m for m in todo if m in have]

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bench_{a.machine}_{mode}_{stamp}.csv")
    cols = ["machine", "mode", "model", "run", "prompt_tokens", "prefill_tok_s", "gen_tokens", "gen_tok_s",
            "first_token_s", "load_s", "model_size_gb", "gpu_pct", "gpu_power_avg_w", "gpu_power_peak_w",
            "parallel_total_tok_s", "parallel_per_req_tok_s", "start_epoch", "end_epoch",
            "ollama", "gpu", "os", "time"]
    summary = []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for model in todo:
            print(f"\n== {model} ==  加载 + 预热…")
            t0 = time.time()
            try:
                r = gen(a.host, model, prompt(0), 16, ctx)
            except urllib.error.HTTPError as e:
                print(f"  加载失败：{e.read().decode()[:300]}"); continue
            load_s = round(r.get("load_duration", 0) / 1e9 or time.time() - t0, 1)
            size, gpu_pct = placement(a.host, model)
            print(f"  加载 {load_s} 秒 · 占用 {size} GB · 在显卡上 {gpu_pct}%")
            gens, pres = [], []
            for i in range(1, a.runs + 1):
                s = GpuSampler(); s.start()
                t_run0 = time.time()                    # 和 tapo_power.py 的記錄對齊用
                ptot = pper = ""
                if a.parallel:
                    ptot, pper, _, _ = parallel_round(a.host, model, a.parallel, a.gen, i)
                    r = gen(a.host, model, prompt(f"{i}-solo"), 16, ctx)   # 占位：并发模式不看单次数字
                elif a.long:
                    r = gen(a.host, model, long_prompt(i), 128, ctx)
                else:
                    r = gen(a.host, model, prompt(i), a.gen, ctx)
                avg, peak = s.stop()
                t_run1 = time.time()
                pt, pd = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 1) / 1e9
                et, ed = r.get("eval_count", 0), r.get("eval_duration", 1) / 1e9
                row = {"machine": a.machine, "mode": mode, "model": model, "run": i,
                       "parallel_total_tok_s": ptot, "parallel_per_req_tok_s": pper,
                       "start_epoch": round(t_run0, 1), "end_epoch": round(t_run1, 1),
                       "prompt_tokens": pt, "prefill_tok_s": round(pt / pd, 1),
                       "gen_tokens": et, "gen_tok_s": round(et / ed, 1),
                       "first_token_s": round(pd, 2), "load_s": load_s,
                       "model_size_gb": size, "gpu_pct": gpu_pct,
                       "gpu_power_avg_w": avg, "gpu_power_peak_w": peak,
                       "ollama": ver, "gpu": info.get("gpu", ""), "os": info["os"],
                       "time": datetime.now().isoformat(timespec="seconds")}
                w.writerow(row); f.flush()
                if a.parallel:
                    gens.append(ptot); pres.append(pper)
                    print(f"  第 {i} 次  {a.parallel} 个请求同时跑：总吞吐 {ptot} tok/s · 单个请求 {pper} tok/s")
                else:
                    gens.append(row["gen_tok_s"]); pres.append(row["prefill_tok_s"])
                    print(f"  第 {i} 次  读提示 {row['prefill_tok_s']} tok/s（{pt} token） · 生成 {row['gen_tok_s']} tok/s（{et} token） · 首字 {row['first_token_s']} 秒")
            summary.append((model, statistics.median(pres), statistics.median(gens), gpu_pct))
            unload(a.host, model)

    print(f"\n汇总（中位数） · {a.machine} · {mode}")
    if a.parallel:
        print(f"{'模型':40}{'单个请求 tok/s':>14}{'总吞吐 tok/s':>12}{'显卡占比':>10}")
    else:
        print(f"{'模型':40}{'读提示 tok/s':>14}{'生成 tok/s':>12}{'显卡占比':>10}")
    for m, p, g, pct in summary:
        print(f"{m:40}{p:>14.1f}{g:>12.1f}{str(pct)+'%':>10}")
    print(f"\n结果已写入：{path}")
    print("下一步：跑 --soak；结束后用 bench_power_merge.py 把 tapo_power.py 的插座功率对进来。")

if __name__ == "__main__":
    main()

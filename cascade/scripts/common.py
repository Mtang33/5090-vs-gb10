"""對比測試的公共部分：節點列表、在節點上跑命令、查 DeepSeek 狀態。在 dgx-head 上運行。

兩個可選開關，默認不影響原來的行為：
  FABRIC_NODES="localhost spark-worker spark-3"   只測這幾台（順序 = rank 順序）
  hosts_override.json（和本文件同目錄）             SSH 改走別的地址，例如局域網斷了臨時走 WiFi：
    {"spark-worker": {"HostName": "<LAN-IP>", "HostKeyAlias": "<LAN-IP>"}, ...}
    HostKeyAlias 填原來 known_hosts 裡記錄的名字，主機密鑰照常校驗。
"""
import json
import os
import subprocess
import urllib.request

# 順序 = 服務的 rank 順序（head 是 rank 0，workers 按 .env.tp4 的 WORKER_HOSTS）
NODES = os.environ.get("FABRIC_NODES", "localhost spark-msi spark-worker spark-3").split()
HEAD_LAN_IP = os.environ.get("FABRIC_MASTER_ADDR", "<LAN-IP>")
_OVR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hosts_override.json")
HOST_OPTS = json.load(open(_OVR)) if os.path.exists(_OVR) else {}


def ssh_argv(h, cmd, extra=()):
    """到節點 h 的 ssh 命令行（帶 hosts_override.json 裡的選項）。"""
    a = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", *extra]
    for k, v in HOST_OPTS.get(h, {}).items():
        a += ["-o", f"{k}={v}"]
    return a + [h, cmd]
DS_URL = __import__("os").environ.get("L3_URL", "http://127.0.0.1:8888")  # GLM 等 vLLM 服務：L3_URL=http://127.0.0.1:8000
CONTAINER = {"localhost": "dsv41-head"}  # 其他節點都是 dsv41-worker


def container(h):
    return CONTAINER.get(h, "dsv41-worker")


def sh(h, cmd, timeout=120, stdin=None, check=False):
    """在節點上跑 shell 命令，返回 (退出碼, stdout, stderr)。"""
    argv = ["bash", "-c", cmd] if h == "localhost" else ssh_argv(h, cmd)
    r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{h}: {cmd[:80]} → {r.returncode}: {r.stderr[-300:]}")
    return r.returncode, r.stdout, r.stderr


def py(h, code, timeout=60):
    """把一段 python 送到節點上跑（python3 -），它最後 print 一個 JSON，返回解析結果。"""
    rc, out, err = sh(h, "python3 -", timeout=timeout, stdin=code)
    if rc:
        raise RuntimeError(f"{h}: python 退出 {rc}: {err[-400:]}")
    return json.loads(out.strip().splitlines()[-1])


def ds_load():
    """(請求數, 在用 token)；服務不在返回 None。分塊 prefill 期間 num_reqs 一直是 0，要看 num_tokens。"""
    try:
        if __import__("os").environ.get("L3_VLLM"):   # vLLM 沒有 /get_load，用 /metrics 的請求數（拿不到在用 token，回 0）
            with urllib.request.urlopen(DS_URL + "/metrics", timeout=5) as r:
                n = sum(float(l.split()[-1]) for l in r.read().decode().splitlines()
                        if l.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")))
            return int(n), 0
        # 優先看 /metrics（要 --enable-metrics）：dsv4.1 分支開了 --sleep-on-idle 後，/get_load 在空閒時
        # 一直停在最後一批的數字（2026-11-01 B4-0 啟動熱身後報 16 個請求），/metrics 是準的
        with urllib.request.urlopen(DS_URL + "/metrics", timeout=5) as r:
            m = {}
            for l in r.read().decode().splitlines():
                if l.startswith(("sglang:num_running_reqs{", "sglang:num_queue_reqs{", "sglang:num_used_tokens{")):
                    k = l.split("{", 1)[0]
                    m[k] = m.get(k, 0) + float(l.split()[-1])
        if "sglang:num_running_reqs" in m:
            return (int(m["sglang:num_running_reqs"] + m.get("sglang:num_queue_reqs", 0)),
                    int(m.get("sglang:num_used_tokens", 0)))
        with urllib.request.urlopen(DS_URL + "/get_load", timeout=5) as r:
            d = json.load(r)[0]
        return d["num_reqs"] + d["num_waiting_reqs"], d["num_tokens"]
    except Exception:
        return None


def ds_running():
    return ds_load() is not None


def ds_containers():
    out = {}
    for h in NODES:
        _, o, _ = sh(h, "docker ps --filter name=dsv41 --format '{{.Names}}'")
        out[h] = o.strip()
    return out

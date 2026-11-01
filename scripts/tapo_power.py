#!/usr/bin/env python3
"""Tapo 智能插頭功率記錄：每秒讀一次，寫 CSV（時間、機器、瓦數），跑分期間一直開著。

python3 工具/tapo_power.py --plug 5090=192.168.4.29 --plug GB10=192.168.4.56
→ 工具/power_<日期時間>.csv；Ctrl-C 結束。跑分結束後用 bench_power_merge.py 對齊。

帳號：TP-Link（Tapo app 登入用的那個）。程式不收密碼參數，只從這兩處讀，由你自己設定：
  1. macOS 鑰匙圈（建議）：security add-generic-password -s tapo -a 你的信箱 -w
     （-w 後面不寫，會提示輸入密碼，不留在命令歷史裡）
  2. 環境變數 TAPO_USERNAME / TAPO_PASSWORD
需要：pip install python-kasa
"""
import argparse, asyncio, csv, os, subprocess, sys, time
from datetime import datetime


KEYCHAIN = [("find-generic-password", "tapo"), ("find-generic-password", "tplinkcloud.com"),
            ("find-internet-password", "tplinkcloud.com")]   # 一般密碼或網站登入密碼都找


def credentials():
    user, pw = os.environ.get("TAPO_USERNAME"), os.environ.get("TAPO_PASSWORD")
    if user and pw:
        return user, pw
    for kind, svc in KEYCHAIN:
        out = subprocess.run(["security", kind, "-s", svc], capture_output=True, text=True)
        acct = next((l.split('"')[-2] for l in out.stdout.splitlines() if '"acct"' in l), None)
        if not acct:
            continue
        # 第一次讀時 macOS 會彈窗問是否允許，點「允許」或「永遠允許」
        pw = subprocess.run(["security", kind, "-s", svc, "-w"], capture_output=True, text=True).stdout.strip()
        if pw:
            return acct, pw
    sys.exit("找不到 TP-Link 帳號：鑰匙圈裡要有 tapo 或 tplinkcloud.com 這筆（一般密碼或網站密碼），"
             "或設 TAPO_USERNAME / TAPO_PASSWORD。")


async def connect(ip, user, pw):
    from kasa import Credentials, Discover
    dev = await Discover.discover_single(ip, credentials=Credentials(user, pw))
    await dev.update()
    return dev


def watts(dev):
    em = dev.modules.get("Energy") if hasattr(dev, "modules") else None
    if em is not None:
        return float(em.current_consumption)
    return float(dev.emeter_realtime.power)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plug", action="append", required=True, help="名稱=IP，例如 5090=192.168.4.29")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  f"power_{datetime.now():%Y%m%d_%H%M}.csv"))
    a = ap.parse_args()
    user, pw = credentials()
    plugs = {}
    for spec in a.plug:
        name, ip = spec.split("=", 1)
        dev = await connect(ip, user, pw)
        print(f"{name}: {ip} · {dev.alias} · {dev.model} · 現在 {watts(dev):.1f} W")
        plugs[name] = dev
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time", "epoch", "machine", "watts"])
        print(f"記錄中 → {a.out}（Ctrl-C 結束）")
        n = 0
        while True:
            t0 = time.time()
            for name, dev in plugs.items():
                try:
                    await dev.update()
                    w.writerow([datetime.fromtimestamp(t0).isoformat(timespec="seconds"), f"{t0:.1f}", name,
                                f"{watts(dev):.1f}"])
                except Exception as e:                  # 插頭偶爾掉線：記一筆空值，下一秒再試
                    w.writerow([datetime.fromtimestamp(t0).isoformat(timespec="seconds"), f"{t0:.1f}", name, ""])
                    print(f"{name} 讀取失敗：{e}")
            f.flush()
            n += 1
            if n % 60 == 0:
                print(datetime.now().strftime("%H:%M:%S"), " · ".join(f"{k} {watts(d):.0f} W" for k, d in plugs.items()))
            await asyncio.sleep(max(0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已停止")

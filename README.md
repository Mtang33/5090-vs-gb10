# RTX 5090 vs DGX Spark (GB10)：跑分原始數據和腳本

湯包實驗室〈5090 vs GB10〉那期影片用到的全部原始數據和跑分腳本。手上有這兩台機器的話，可以自己跑一遍對答案。

> 〈一張 5090，打得過四台 GB10 嗎？〉那期的大模型、交換器、耗電數據在 [`cascade/`](cascade/)。

## 機器

| | 5090 | GB10 |
| --- | --- | --- |
| 整機 | XIDAX 預裝機：Ryzen 7 9800X3D、64 GB、RTX 5090 32 GB | NVIDIA DGX Spark：GB10、128 GB 統一記憶體 |
| 系統 | Ubuntu 26.04、驅動 595.99.02 | DGX OS、驅動 580.178.04 |
| 量電 | Tapo 智能插座，只接這一台 | Tapo 智能插座，4 台 DGX + 交換機一起（只看「增量」） |

電量一律用**插座增量**：測試時的平均功率減掉這個插座開跑前的待機功率。GB10 那個插座上有 4 台機器，所以不比待機，只比多用了多少電。

## 測了什麼

| 項目 | 腳本 | 數據 |
| --- | --- | --- |
| LLM 三組跑分（短提示、長提示、4 路並發），Ollama 0.34.4 | `scripts/bench_5090_gb10.py` | `data/bench_*_{short,long,parallel4}_*.csv` |
| qwen3.6:35b 補測（Ollama 0.35.0，num_batch 2048） | 同上（`BENCH_NUM_BATCH`） | `data/qwen3.6補測_0.35/` |
| gpt-oss:120b 滿載 1 小時 | 同上（`--soak 3600`） | `data/滿載1小時/` |
| gpt-oss:120b 換 llama.cpp、調大批次 | `scripts/bench_llamacpp.py` | `data/5090_llamacpp/`、`data/GB10_llamacpp/` |
| 實戰：一條粘土動畫 short（Qwen-Image 關鍵幀、Wan 2.2 鏡頭、MiniMax H3）、Qwen-Image-Edit 修圖、Remotion 渲染；三組：1 張 5090 / 1 台 GB10 / 4 台 GB10 並行 | `scripts/bench_comfy.py`、`scripts/qedit.py`、`workflows/` | `data/實戰{A,B,C}組_*` |
| 字幕校對（qwen3:30b-a3b-instruct，E02 字幕） | `scripts/bench_caption.py` | `data/實戰A組_5090/`、`data/實戰B組_GB10/caption/` |
| 畫質（同 seed 的 PSNR 和並排圖） | `scripts/bench_psnr.py` | `data/實戰_畫質比對/` |
| FunASR 轉寫（paraformer-zh + VAD + 標點） | `scripts/bench_asr.py` | `data/FunASR/` |
| 插座功率記錄與對齊 | `scripts/tapo_power.py`、`scripts/bench_power_merge.py`、`scripts/實戰_功耗匯總.py` | 各資料夾裡的 `power_*.csv`、`*_插座功率.csv` |

## 主要結果

| | 5090 | GB10 |
| --- | --- | --- |
| qwen3:30b-a3b 生成 | 335.8 tok/s | 86.0 tok/s |
| gpt-oss:120b 生成（Ollama） | 45.9 tok/s | 43.1 tok/s |
| gpt-oss:120b 讀 3 萬 token 長提示，首字（Ollama） | 63 秒 | 16 秒 |
| gpt-oss:120b 兩邊都用 llama.cpp 調好：生成 / 首字 / 每份長提示用電 | 43.7 tok/s / 14.8 秒 / 1.14 Wh | 53.1 tok/s / 13.4 秒 / 0.49 Wh |
| gpt-oss:120b 滿載 1 小時：每千 token 用電 | 1.84 Wh | 0.76 Wh |
| 一條粘土 short（含載入） | 16.8 分 / 150 Wh | 1 台 54.7 分 / 99 Wh；4 台並行 16.7 分 / 114 Wh |
| FunASR 轉 7.7 分鐘錄音 | 0.93 秒 | 2.18 秒 |

## 怎麼自己跑

```bash
python3 scripts/bench_5090_gb10.py --machine 5090 --host http://127.0.0.1:11434 --models gpt-oss:20b
python3 scripts/bench_5090_gb10.py --machine GB10 --host http://127.0.0.1:11434 --soak 3600 --soak-model gpt-oss:120b
python3 scripts/bench_comfy.py --machine 5090 keys workflows/clay_scenes.json
```

各腳本開頭的說明寫了完整用法。ComfyUI 是 commit e9027f2；Wan 跑在開 SageAttention 的實例，Qwen-Image 跑在不開的實例。量電需要 Tapo 插座和 `pip install python-kasa`，帳號只從環境變數或 macOS 鑰匙圈讀，腳本裡沒有任何帳密。

## 注意

- 實戰對比的關鍵幀圖片（`workflows/` 裡 `--keys` 用的那組）沒有放進來，需要的話先用 `bench_comfy.py keys` 自己生成。
- 不同架構（5090 是 sm120，GB10 是 sm121）同 seed 出圖不會逐像素相同，構圖一致、細節不同；同一種機器之間逐幀相同。
- 數據按原樣公開，CSV 第一列是欄位名。

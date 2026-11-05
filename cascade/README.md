# 四台 GB10 級聯：大模型、交換器、耗電的原始數據和腳本

湯包實驗室〈一張 5090，打得過四台 GB10 嗎？〉那期影片用到的數據。黏土短片那部分（1 張 5090 / 1 台 GB10 / 4 台 GB10 並行）在上一層的 `data/實戰{A,B,C}組_*`，這裡只放四台一起跑大模型、環形 vs 交換器、耗電的部分。

## 機器和接法

- 四台 DGX Spark（GB10，每台 128 GB 統一記憶體），經 MikroTik CRS804 交換器相連：每兩台之間 111.6 Gb/s、延遲 2.5 µs。
- 「環形」是之前四台首尾相連、不經交換器的接法。
- 量電：Tapo 智慧插座，四台 DGX + 交換器共用一個插座，只看「增量」（測試時的平均功率減掉待機功率）。

## 有什麼

| 資料夾 | 內容 |
| --- | --- |
| `data/serving/ring_A-msi-drain/` | DeepSeek V4.1 Flash，環形接法（基線） |
| `data/serving/switch_B2/` | 同一版推論軟體，只換成交換器 |
| `data/serving/switch_B4-2/` | 交換器 + 新版推論映像和核心；`l4_stability_1h.json` 是 1 小時穩定性（16 路解碼 + 每 10 分鐘 13.1 萬 token needle） |
| `data/serving/switch_B4-2u_production/` | 現在日常用的設定（B4-2 + 循環中止時回傳截斷結果） |
| `data/serving/Nemotron-first-try/` | Nemotron 3 Ultra 550B 第一次：照官方設定，讀 3.2 萬 token 時記憶體耗盡 |
| `data/serving/Nemotron-lowmem/` | 調低記憶體後重試（顯存佔比 0.85→0.80、單批 16384→8192、上下文 262k→131k）；`mem_min.txt` 是各台最低剩餘記憶體 |
| `data/e02_eval/` | 三個模型做同一份 E02 逐字稿的四個活：標刪字、字幕校對、章節和標題、長文問答。只有分數、耗時和 token 數，**模型的原始輸出沒有放**（輸入含沒剪進成片的錄音轉寫） |
| `data/power/` | 四台跑 DeepSeek 時的插座逐秒功率，`marks_20261105.csv` 是每段負載的起止時間（epoch 秒） |
| `scripts/l3_serving.py` | 服務級測試：解碼（prose / code，單路和 16 路）、首字時間、長輸入 prefill + needle、批量長輸出 |
| `scripts/run_eval.py` | E02 四個活的提示和自動評分（輸入檔 `inputs.json` 沒有公開，原因同上） |
| `scripts/cascade_power_ds.sh` | 耗電對齊用的負載：待機 5 分 → E02 四題 ×3 → L3 quick → 待機 3 分 |

## 主要結果（token／秒）

| | 環形 | 交換器 | 交換器 + 新版軟體（現在） |
| --- | --- | --- | --- |
| DeepSeek 讀長稿子 3.2 萬 / 13.1 萬 / 25.9 萬 token | 3,091 / 2,886 / 2,668 | 3,425 / 3,139 / 2,849 | 4,830 / 4,766 / 4,458 |
| DeepSeek 單人寫程式 | 66.4 | 59.3 | 102.7 |
| DeepSeek 單人寫文章 | — | 持平 | 58.6 |

- 強制 NCCL 用 Ring 演算法：寫文章單路慢 24%；交給它自己選才是上表的結果。
- Nemotron 3 Ultra（調低記憶體後）：寫文章 12.3、寫程式 19.8；讀 3.2 萬 / 6.4 萬 / 12 萬 token：671 / 577 / 516，needle 全對。
- E02 標刪字（165 句、我手剪刪 61 句，一刀不剪的吻合度 63%）：DeepSeek 75.2%、GLM-5.3-Flash 73.9%、Nemotron 49.7%；九千字長文問答三個都是 10／10。我自己盲評：字幕校對 DeepSeek > GLM > Nemotron，章節和標題 Nemotron > GLM > DeepSeek。
- 耗電：DeepSeek 做完一遍 E02 四個活約 15 秒，比待機（約 266 W）多 +275–286 W，每遍多用 1.10–1.16 Wh。插座每秒只讀一次，精度約 ±10%。

## 注意

- 三個模型用的推論引擎和量化不同（DeepSeek：SGLang、FP8；GLM-5.3-Flash 和 Nemotron：vLLM、NVFP4），數字只代表在這四台上跑多快，不代表模型本身的好壞。
- GLM-5.3-Flash 的速度數據沒有公開：測的時候帶了一個授權不允許商用的草稿模型。
- `scripts/common.py` 裡的區網位址換成了 `<LAN-IP>`，要跑請用環境變數 `FABRIC_MASTER_ADDR` 等設定自己的機器。
- 數據按原樣公開。

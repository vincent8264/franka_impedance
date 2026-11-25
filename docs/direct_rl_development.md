# Franka Panda Direct RL 開發紀錄

**最後更新：2026-11-25｜ECE 276C**

> **2026-11-25 程式修正與重跑：** `scripted_actions` 已改用目前夾爪映射的 +1／−1 動作，`validate_grasp_physics.py` 的 `applied_effort` 改為記錄環境實際送出的力。已用下方「2026-11-25 名目驗證」設定重跑；其餘 CSV、GIF 和策略成功率保留原執行結果。

## 2026-11-25 名目驗證

從 `C:\Users\user\IsaacLab` 的固定 commit `e78aa3e` 執行，Newton／CUDA、120 Hz、種子 42、單環境。`uv run --locked --no-sync --offline` 使用現有虛擬環境與已下載資產。更新後的腳本在 `lift` 第 172 個索引步通過幾何及雙指接觸成功條件；含 36 步靜置共 659 步。CSV 的夾爪 `applied_effort_7` 由環境實際命令擷取：閉合首步及抬升為 −17.5 N；張開階段受到行程保護，實際為 0–0.783 N。[腳本 CSV](../logs/direct_grasp/validation-20261125/scripted.csv) 為本機忽略的輸出。

同設定下，新的 Cartesian impedance baseline 完成 1/1 回合，幾何及接觸驗證成功率皆為 100%，於第 659 步終止；方塊底部最大離桌 15.75 cm，高於 10 cm 且雙指接觸各超過 0.5 N 共 60 步。[評估 JSON](../logs/impedance/validation-20261125.json)及[逐步 CSV](../logs/impedance/validation-20261125.csv)為本機忽略的輸出。這只驗證單一名目回合；夾爪左右指最大位置約 40.028／40.034 mm，較 40 mm 上限各超出 0.028／0.034 mm，仍高於下文的每指 0.02 mm 驗收門檻。

## 目前進度

- **已完成：** PPO 先導訓練 16 次更新，續訓 72 次，共 **88 次更新、33,792 筆環境轉移**（先導 6,144 筆、續訓 27,648 筆）。
- **續訓驗證：** 先導 `final.pt` 與續訓開始前的 `initial.pt` 完全相符，包含 actor／critic 權重、Adam 最佳化器狀態，以及兩者觀測正規化的均值、變異數、標準差與樣本數。
- **目前結果：** 兩個 checkpoint 在固定條件下各評估 16 回合，幾何成功與接觸驗證成功皆為 0/16。各追加一回合逐步診斷，兩者皆在接近階段失敗，未碰到方塊。

## 系統設定

| 項目 | 目前設定 |
| --- | --- |
| 模擬與訓練 | Isaac Lab（固定 commit `e78aa3e`）+ Newton / CUDA + RSL-RL PPO 5.4.1 |
| Observation | **46 維**：關節、夾爪、末端與方塊狀態 |
| Action | **8 維**：7 個關節力矩 + 1 個夾爪施力；120 Hz |
| 控制方式 | RL 決定動作；手臂力矩先縮放至上限的 **10%**，加上重力補償後再限幅；**沒有 PD 姿態維持或 Impedance Controller** |
| 夾爪 | 第 8 維只對左指施力，右指隨耦合運動。−1／0／+1 的名目力為 −17.5／0／+1.05 N（閉合／零力／張開）。按兩指中較小的預估剩餘行程降力，向前估計兩步，最後 2 mm 線性降至 0 N。 |
| 方塊 | 質量 **0.2 kg**；靜、動摩擦係數均為 **0.5** |
| 成功條件 | 方塊最低點高於桌面 **10 cm**，持續 **0.5 秒**；每回合最多 **12 秒** |
| Reward | 接近、雙指接觸、抬升、完成獎勵與動作懲罰；接觸版使用 **0.5 N** 門檻，參數仍待驗證 |

## 主要程式

| 檔案 | 用途 |
| --- | --- |
| `direct_rl_env.py` | 串接場景、Observation、Action、Reward 與 Reset |
| `grasp_rewards.py` | 計算 RL Reward |
| `train_direct_rl.py` | 啟動 Rollout、Contact 測試與 PPO 訓練 |
| `smoke_direct_rl.py` | 驗證控制、重力補償、接觸和重置 |
| `validate_grasp_physics.py` | 以 Direct RL 動作介面執行抓取、抬升和維持，逐步記錄力、飽和與回報 |
| `eval_direct_rl.py`、`grasp_diagnostics.py` | 評估 Checkpoint；選用 `--diagnostics` 記錄重置前的逐步資料 |
| `grasp_gif.py` | 將模擬位置資料轉成 GIF |
| `grasp_task.py`（共用） | Observation、重置、抬升高度判定 |

## 測試結果

| 測試 | 結果 |
| --- | --- |
| 場景與環境 | Franka、桌子、方塊正常載入；多環境的 Action / Observation / Reward 數值與形狀正常 |
| Reset / Success | 腳本設定方塊高度後，**60 步**觸發成功；逾時與各環境獨立重置正常（**不是機器人自行抓取成功**） |
| 重力補償 | 0.2 秒零動作測試，最大關節漂移由 **0.237 rad** 降至低於 **0.000001 rad** |
| 接觸感測 | 可區分雙指接觸方塊、接觸桌面與無接觸；自然抓取時的穩定性待測 |
| PPO 短測試 | 4 個環境共 **96 筆資料**，完成 1 次更新並儲存 Checkpoint |
| Checkpoint 評估 | 單台 **0/1**、四台 **0/4**；平均 Episode Return 約 **−0.0546** |
| GIF | `single.gif` 與 `parallel.gif` 各 **40 幀**，皆可讀取 |

既有 `single.gif` 與 `parallel.gif` 為前 2 秒的姿態示意動畫。下方兩份診斷 GIF 涵蓋完整 12 秒。

## PPO 策略診斷

兩個 checkpoint 各用新場景執行一次確定性推論。名目初始條件、種子 42、120 Hz、單環境；兩回合皆在第 1,440 步逾時。CSV 含重置前最後一步。GIF 採相同視角與 20 fps，各 240 幀，已逐幀解碼。

| 實測指標 | 16 次更新 | 88 次更新 | 腳本基準 |
| --- | ---: | ---: | ---: |
| 最小手部原點／夾取中心距離（cm） | 20.51／13.92（第 1 步） | 18.12／9.77（8.65 s） | 9.51／未記錄 |
| 左／右指對方塊峰值（N） | 0／0 | 0／0 | 20.10／17.29 |
| 雙指同時 > 0.5 N（步） | 0 | 0 | 277（有力值的步） |
| 方塊落桌後最大底部淨高（cm） | ≈0 | ≈0 | 14.49 |
| 幾何／接觸驗證成功 | 否／否 | 否／否 | 是／是 |
| 接近獎勵／動作懲罰總量 | 0.001526／0.003254 | 0.006682／0.005953 | 0.019440／0.000944 |

**實測。** 16 次更新從起始最近位置偏離；88 次更新曾靠得更近，但全程未碰到方塊。兩者都停在接近階段，接觸、抬升、完成獎勵皆為 0。夾爪張開指令分別佔 94.0% 與 98.2%。手臂正規化輸出分別限幅 1、5 次，加重力補償後的馬達命令均未再限幅。腳本先靜置 36 步，接著以同一動作介面抓取，於 `lift` 第 179 個索引步通過持續 0.5 秒的成功判定。基準 CSV 末列保留成功旗標，但無末步力與高度。

**待驗證的假設。** 接近獎勵使用手部原點距離，可能弱化對夾取中心與姿態的引導；接觸前只有接近獎勵與動作懲罰。腳本在相同 10% 手臂動作尺度下成功，顯示名目條件下可行，仍無法排除尺度影響探索。`gamma=0.99` 在 120 Hz 下使 1 秒後獎勵權重降至約 0.30，可能削弱延遲的抬升訊號。88 次更新的近距離表現改善，訓練不足仍是可能原因；本次單回合比較無法判定因果。

- 逐步紀錄：[16 次 CSV](../logs/direct_grasp/diagnostics-20261123/ppo_u016.csv)、[88 次 CSV](../logs/direct_grasp/diagnostics-20261123/ppo_u088.csv)、[腳本基準](../logs/direct_grasp/physics-20261123/scripted.csv)。
- 示意動畫：[ppo_u016.gif](../logs/direct_grasp/gif/ppo_u016.gif)、[ppo_u088.gif](../logs/direct_grasp/gif/ppo_u088.gif)。
- Checkpoint 分別為 `logs/direct_grasp/20261123-132200-063437/final.pt` 與 `logs/direct_grasp/20261123-154618-376733/final.pt`。各次評估指定 `--num-envs 1 --episodes 1 --episode-seconds 12 --seed 42 --record-steps 1440 --device cuda:0 --viz none`、`--checkpoint`、`--diagnostics` 與 `--gif`。

## 腳本交接診斷（88 次更新）

本回合有腳本協助定位，獨立策略成功率仍為 0/16。種子 42、120 Hz，先靜置 36 步，再沿用成功腳本的接近 200 步、下降 150 步。第 386 步末夾取中心距方塊 **0.88 cm**、指墊內面間隙 **8.00 cm**、手部速度 **2.94 cm/s**，雙指接觸力皆為 0 N。腳本段的共同量測與動作逐步吻合既有基準。第 387 步起由凍結策略接手，直接使用當下觀測；共執行 1,054 個策略步，12 秒逾時。

| 比較 | 雙指同時 > 0.5 N | 方塊底部最大淨高 | 結果 |
| --- | ---: | ---: | --- |
| 88 次更新，完整策略回放 | 0 步 | 落桌後約 0 | 最小夾取中心距離 9.77 cm，未接觸 |
| 88 次更新，腳本交接後 | 1 步 | 0.63 cm | 有短暫接觸，未形成可抬升的抓握 |
| 完整腳本 | 277 個有力值的步 | 14.49 cm | 通過 0.5 秒持握判定 |

**策略接手後。** 第一步施加 +70 N 張開力。第 394 步施加 −35.46 N，雙指接觸力達 31.59／33.85 N，下一步即失去雙指接觸。策略有 1,049／1,054 步下達張開指令，夾爪動作共限幅 554 次。接觸獎勵總量為 0.00025，抬升與完成獎勵均為 0。

**夾爪寬度驗證。** 觀測 `width_m` 是兩個滑動關節座標的有符號和，資產中每指範圍皆為 0–4 cm，且各自的正方向皆為張開。原 88 次回放有 169 步為負；交接策略段有 293 步為負，這些步的指墊內面投影間隙也都為負。第 886 步兩關節為 −7.49／−7.52 cm，指墊面有符號間隙為 −15.01 cm，兩指墊中心的幾何距離仍為正值 13.49 cm。負觀測對應模擬關節越界及指墊順序反轉，超出有效物理開口範圍。

同一腳本預抓姿態的夾爪力脈衝結果見下節。

- [交接 CSV](../logs/direct_grasp/diagnostics-20261123/ppo_u088_handoff.csv) 以 `phase` 標示控制來源，包含重置前末步、獎勵分項與限幅前後馬達命令。[交接 GIF](../logs/direct_grasp/gif/ppo_u088_handoff.gif) 為完整 12 秒，20 fps、240 幀，已逐幀解碼。
- 從相鄰 `IsaacLab/` 以既有虛擬環境執行 `../franka_impedance/scripts/diagnose_grasp_handoff.py`，傳入上列 88 次 checkpoint 的 `--checkpoint`、上列輸出路徑的 `--csv` 與 `--gif`，以及 `--device cuda:0 --viz none`。

## 夾爪限位診斷（2026-11-24）

設定記錄顯示，兩指的直線關節位置限位皆為 **0–40 mm**，左／右指的 effort 上限分別為 **70／1 N**。Direct RL 只對耦合夾爪的左指送出有號力命令，修正前第 8 維可輸出 ±70 N。力上限限制命令大小，位置限位則由物理約束處理。兩指致動器的剛度與阻尼均為 0，Newton 原生致動器已停用。

各案例從相同腳本預抓姿態開始，以 120 Hz 施力 60 步。脈衝前左指已超過上限約 0.025 mm。下表列出相對於位置限位的最大越界量。資料見 [設定 JSON](../logs/direct_grasp/gripper-limits-20261124/effort_config.json) 與 [逐步 CSV](../logs/direct_grasp/gripper-limits-20261124/effort_pulses.csv)。

| 力命令 | 方塊 | 左指最大越界（mm） | 右指最大越界（mm） |
| --- | --- | ---: | ---: |
| +1.05 N（腳本張開） | 原位 | 0.035 | 0.007 |
| −17.5 N（腳本閉合） | 原位 | 0 | 0 |
| +70 N | 原位 | 39.16 | 32.06 |
| −70 N | 原位 | 0 | 0 |
| +70 N | 移開 | 88.48 | 85.41 |
| −70 N | 移開 | 73.85 | 70.61 |

移開方塊後，高力脈衝仍造成越界。力命令限幅只限制輸入力，兩指的位置限位由求解器處理。限位啟用時序見下方零施力診斷。

## Direct RL 夾爪修正與驗證（2026-11-24）

第 8 維改為分向線性縮放，−1／0／+1 對應 −17.5／0／+1.05 N。每步以兩指的位置與朝命令端點的速度預估兩個物理步後的剩餘行程，取較小值。在預估剩餘 2 mm 內線性降力，至端點歸零，反向退離仍可施力。此保護只減少原方向的力。七個手臂扭矩通道與重力補償維持原設定。

兩輪皆為 Newton、CUDA、120 Hz、種子 42。檔案時間顯示，13:13 的 `*-position-only.csv` 為初版，13:14 加入速度保護，13:16 的 `empty.csv`、`scripted.csv` 為最新版。兩輪各完成 330 步空夾爪測試及 623 步腳本，腳本均正常成功終止。下表為左／右指最大位置越界（mm）。[初版空夾爪](../logs/direct_grasp/gripper-fix-20261124/empty-position-only.csv) · [初版腳本](../logs/direct_grasp/gripper-fix-20261124/scripted-position-only.csv) · [最新版空夾爪](../logs/direct_grasp/gripper-fix-20261124/empty.csv) · [最新版腳本](../logs/direct_grasp/gripper-fix-20261124/scripted.csv)

| 測試 | 初版：僅位置 | 最新版：位置與速度 |
| --- | ---: | ---: |
| 空夾爪在上限維持張開 | 0.032622／0.020456 | 0／0.000045 |
| 空夾爪全幅閉合至下限 | 16.430／13.997 | 6.766／5.201 |
| 空夾爪由下限重開至上限 | 2.986／2.640 | 2.999／2.765 |
| 空夾爪重開後零力 | 0／0 | 0.013079／0.009682 |
| 腳本張開、接近及下降 | 0.094570／0.044696 | 0.028457／0.034440 |
| 腳本閉合、抬升及持握 | 0／0 | 0／0 |

**初版。** 腳本抓取成功，但空夾爪閉合越界達 16.43 mm，限位保護未通過。

**最新版。** 腳本接觸驗證成功。閉合階段雙指峰值力為 31.98／21.38 N，抬升期間約為 8.72–8.75／8.75–8.78 N。方塊離桌 15.74 cm，並在超過 10 cm 時維持雙指接觸 60 步（0.50 秒）。空夾爪兩端仍有回彈與振盪。指墊面有號間隙在閉合瞬間最低 −12.01 mm，重開最高 85.72 mm。閉合與重開的最大越界當步，施力已為 0 N。測試末端能從接近零間隙重開到 79.70 mm，未見卡死。兩份最新 CSV 的數值均為有限值。

本次驗證以 `probe.py` 重用既有腳本的手臂軌跡，改用 +1／−1 命令維持原腳本的名目張開／閉合力，並記錄保護後的實際力。舊驗證程式中的夾爪動作值及力換算沿用原映射。[本次驗證程式](../logs/direct_grasp/gripper-fix-20261124/probe.py)

## 零施力限位診斷（2026-11-24）

**已驗證。** 移開方塊，夾爪全程 0 N。兩指從距下／上限 1 mm 處靜止起步，8 步內均在行程內。給定朝下限 −1.80 m/s 或朝上限 +0.53 m/s 的初速，首步分別越界 14.00／3.42 mm。首步均無手指限位列，第 2 步才出現兩列。首步位移與初速乘上 8.33 ms 相符。[零施力 CSV](../logs/direct_grasp/gripper-fix-20261124/zero-effort-limit-probe.csv)

兩指的限位已啟用，實際 `margin` 均為 0。MuJoCo Warp 在積分前，以當前位置到最近限位的距離判定是否小於 `margin`，再建立限位列。速度在限位列建立後才進入該列的響應計算。步首仍距限位 1 mm 時，即使該步會越界也沒有可施加反力的限位列。積分越界後，下一步才符合啟用條件。[MuJoCo 限位與積分](https://mujoco.readthedocs.io/en/stable/computation.html#limit)

| 暫時對照設定 | 零施力結果 |
| --- | --- |
| 步長縮為 2.08 ms，`margin=0` | 首步越界降至 2.75／0.104 mm，仍於第 2 步啟用。跨界步全場接觸列為 0。 |
| 雙指 `margin=2 mm` | 兩種步長均在首步啟用，測試期間無越界。距限位 1 mm 的靜止手指也被推離端點，120 Hz 首步速度約 0.276 m/s。 |
| 僅把 `solref` 改為 (0.035 s, 1.0) | 首步越界量與第 2 步啟用的時序相同，之後越界峰值增加。 |
| 暫停雙指等式耦合 | 等式列由 1 降為 0，仍於首步越界、第 2 步啟用。 |

步長對照直接縮短診斷環境步長，量測期間同為約 66.7 ms。[120 Hz CSV](../logs/direct_grasp/gripper-fix-20261124/limit-sweep-1.csv) · [480 Hz CSV](../logs/direct_grasp/gripper-fix-20261124/limit-sweep-4.csv) · [對照程式](../logs/direct_grasp/gripper-fix-20261124/limit_sweep_temp.py)

**維持 120 Hz 控制的對照。** 只在診斷設定增加 Newton 求解子步。下表為控制步末採樣的左／右指最大越界量。兩種子步設定均完成腳本抓取，方塊離桌約 15.7 cm，且在高於 10 cm 時持續雙指接觸 0.5 秒。

| 求解子步 | 空夾爪閉合（mm） | 空夾爪重開（mm） | 逐步資料 |
| ---: | ---: | ---: | --- |
| 1 | 6.766／5.201 | 2.999／2.765 | [空夾爪](../logs/direct_grasp/gripper-fix-20261124/empty.csv) |
| 4 | 2.514／0.706 | 0.669／0.398 | [空夾爪](../logs/direct_grasp/gripper-fix-20261124/substeps-4-empty.csv) · [腳本](../logs/direct_grasp/gripper-fix-20261124/substeps-4-scripted.csv) |
| 8 | 1.528／0 | 0.378／0.101 | [空夾爪](../logs/direct_grasp/gripper-fix-20261124/substeps-8-empty.csv) · [腳本](../logs/direct_grasp/gripper-fix-20261124/substeps-8-scripted.csv) |

**待驗證。** 限位的柔順響應與後續指墊接觸可能影響回彈及再次越界。越界是否影響既有 PPO 策略的持握仍待確認。

## 子步最終驗收（2026-11-24）

**驗收門檻。** 兩個實際碰撞指墊沿閉合軸各厚 15.20 mm，中心距指根座標原點 7.58 mm，內面位於本地 −0.02 mm。兩指在標稱下限 0 mm 時，內面有號間隙為 −0.04 mm。在此固有重疊之外，另給同量的 0.04 mm 數值容差，平均分配至兩指。每指在任一端的最大越界須 **≤ 0.02 mm**，指墊間隙須 **≥ −0.08 mm**。這是依模擬碰撞幾何訂定的驗收容差。重開須回到標稱開口約 79.96 mm，腳本須在方塊離桌超過 10 cm 時保持雙指接觸力各大於 0.5 N，連續 0.5 秒。[載入後的指墊幾何](../logs/direct_grasp/gripper-fix-20261124/final-substeps-4-geometry.json)

**已驗證。** CUDA 單環境、種子 42，控制維持 120 Hz。4／8 子步的求解步長為 2.083／1.042 ms。每次求解後記錄兩指位置、速度、越界量、限位列及有號指墊間隙，分別取得 3,944／7,888 筆。每個控制步的最後一筆與環境狀態相符。下表為逐子步的左／右指最大越界量。[4 子步 CSV](../logs/direct_grasp/gripper-fix-20261124/final-substeps-4.csv) · [8 子步 CSV](../logs/direct_grasp/gripper-fix-20261124/final-substeps-8.csv) · [診斷程式](../logs/direct_grasp/gripper-fix-20261124/final_substep_probe.py)

| 測試 | 4 子步（mm） | 8 子步（mm） |
| --- | ---: | ---: |
| 零施力，雙端靜止 | 0／0 | 0／0 |
| 零施力，朝下限 −1.80 m/s | 2.750／2.750 | 1.491／1.491 |
| 零施力，朝上限 +0.53 m/s | 0.104／0.104 | 0.375／0.375 |
| 空夾爪全幅閉合 | 2.864／1.129 | 1.959／0.298 |
| 空夾爪重開 | 0.669／0.398 | 0.551／0.344 |
| 腳本全程（峰值發生在接近階段） | 0.042／0.009 | 0.069／0.004 |

零施力仍在跨界後的下一子步出現限位列。朝下限時，兩設定的最小指墊間隙分別為 −5.540／−3.021 mm。空夾爪閉合最小間隙為 −4.032／−2.298 mm，重開後均回到約 79.96 mm。腳本在閉合、抬升與持握階段均無越界，兩設定皆抬升至約 15.7 cm，並通過 0.5 秒雙指持握。[4 子步腳本](../logs/direct_grasp/gripper-fix-20261124/final-substeps-4-scripted-control.csv) · [8 子步腳本](../logs/direct_grasp/gripper-fix-20261124/final-substeps-8-scripted-control.csv)

**效能。** RTX 5060 Laptop GPU 單環境，開啟 CUDA Graph，關閉畫面與逐子步記錄。重播相同的 600 步動作三次，以環境步進時間中位數比較。4 子步為 5.52 秒（109 步／秒），8 子步為 6.17 秒（97 步／秒），耗時增加約 12%。此估計適用於本次單環境步進。[4 子步計時](../logs/direct_grasp/gripper-fix-20261124/final-substeps-4-benchmark.json) · [8 子步計時](../logs/direct_grasp/gripper-fix-20261124/final-substeps-8-benchmark.json)

**最終判定。** 4 與 8 子步均未通過幾何門檻，暫無可採用的最終子步數。8 子步縮小全幅閉合越界，但零施力朝上限的峰值高於 4 子步。保留現行專案設定。正式評估時，Cartesian impedance、Direct RL 與 residual RL 三種控制器須共用同一物理後端、步長、子步數及求解器參數。

## 檔案位置與重現指令

- Checkpoint：`logs/direct_grasp/20261122-155118-338689/final.pt`
- GIF：`logs/direct_grasp/playback-20261123/single.gif`、`parallel.gif`
- 以下指令從相鄰的 **`IsaacLab/`** 資料夾執行；不需要修改 Isaac Lab 原始碼。

```powershell
# PPO 1 次更新（僅短測試）
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/train_direct_rl.py --mode ppo --num-envs 4 --iterations 1 --contact-reward --viz none

# 讀取既有 Checkpoint，評估 4 個環境並儲存 GIF
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/eval_direct_rl.py --checkpoint ../franka_impedance/logs/direct_grasp/20261122-155118-338689/final.pt --episodes 4 --num-envs 4 --gif ../franka_impedance/logs/direct_grasp/playback-20261123/parallel.gif --viz none
```

## Phase 3 — 名目條件抓取物理驗證

在相鄰的 `IsaacLab/` 執行。感測器測試會搬動桌面，因此使用獨立的模擬實例。抓取腳本從名目重置狀態開始，讓方塊自行落到桌面，其後僅透過既有的 8 維力矩／施力動作介面操作機器人。測試腳本選用與初始姿態接近、夾爪軸對齊方塊面的朝下姿態，並以資產的夾爪墊中心規劃夾取位置。

```powershell
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/train_direct_rl.py --mode contact --num-envs 1 --viz none

# 資產伺服器逾時時，使用此電腦已下載的同一份 Isaac 6.1 資產。
$env:ISAACSIM_ASSET_ROOT = (Join-Path $env:LOCALAPPDATA 'Temp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1').Replace('\','/')
$env:WARP_CACHE_PATH = Join-Path $env:TEMP 'franka-direct-grasp-warp'
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/validate_grasp_physics.py --output ../franka_impedance/logs/direct_grasp/physics-20261123/scripted.csv --viz none
```

無接觸時，兩指方塊與桌面通道均為 0 N。方塊接觸 fixture 的左右方塊通道各自峰值為 126.33 和 119.13 N，且兩指曾同時超過 0.5 N。僅接觸桌面時，左右桌面通道為 660.34 和 649.24 N，方塊通道仍為 0 N。這些 fixture 用來驗證通道隔離。

首次腳本啟動在 Jacobian 的 Warp 核心編譯階段遇到預設快取權限錯誤。重跑將快取指向暫存目錄，並使用全新場景隔離移動過桌面的 fixture。

## 開發紀錄

- **2026-11-22：** 完成 Direct RL 環境與四項 Smoke Tests；PPO 成功更新一次。
- **2026-11-23：** 完成 Checkpoint Playback、單台／四台評估及 GIF 輸出；固定條件共 5 回合，成功率 0%。
- **2026-11-23：** 完成 16／88 次更新的逐步診斷及全回合 GIF；兩者均未碰到方塊。
- **2026-11-24：** 完成 88 次更新的腳本交接診斷；測得單步雙指接觸與指墊越界，未抬升。

- **2026-11-24：** 彙整夾爪力脈衝與位置限位結果，記錄 Direct RL 縮放、限位保護及 checkpoint 評估順序。
- **2026-11-24：** 完成 Direct RL 夾爪縮放與速度保護驗證。腳本抓取成功，空夾爪仍有顯著越界，待釐清後再評估 PPO。
- **2026-11-24** 零施力對照確認限位依積分前的位置啟用，並量測步長、限位參數及雙指耦合的影響。
- **2026-11-24** 完成 4／8 子步逐次量測與指墊幾何驗收。兩設定均通過腳本持握，仍有跨界與指墊重疊。

> **後續更新方式：** 更新上方「目前進度、測試結果、下一步」，並在此新增簡短日期紀錄即可，不用重複整篇說明。

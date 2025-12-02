# ===== 系統與通用套件 =====
import os
import re
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import defaultdict
# SciPy 信號處理
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter, welch
# MNE (EEG 處理)
import mne
from mne import create_info
from mne.channels import make_standard_montage
# Scikit-learn - 模型與處理流程
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    ShuffleSplit,
    StratifiedKFold,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin
# 儲存資料
import joblib
from collections import defaultdict
# 資料視覺化
import seaborn as sns
# 其他設定
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import datetime
from collections import Counter

# ===== 載入資料 =====
# Use absolute path relative to this file to avoid CWD issues
base_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(base_dir, '../PKL/S01-06_dataset_2class_wave_fist_1202.pkl')
if not os.path.exists(save_path):
    raise FileNotFoundError(f"找不到資料檔案: {save_path}，請確認路徑與檔案存在。")
norm_mod_data = joblib.load(save_path)  # 我們把資料載入到 norm_mod_data

# ===== 一些全域參數 =====
modalities_list = ["MI", "ME"]
self_sfreq = 250
sfreq = 250

def ezbandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

# ===== 第一部分：資料裁切（保留原程式邏輯） =====
for subject, modalities in norm_mod_data.items():
    print("subject:", subject)
    for modality in modalities_list:
        if modality not in modalities:
            continue
        for date, datas in modalities[modality].items():
            for i in datas:
                data = datas[i]['data']
                run_start = self_sfreq * 15
                run_end = run_start + self_sfreq * 480
                # 裁切資料
                data = data[:, run_start:run_end]
                # 驗證資料形狀並更新
                if data.shape == (3, 120000):
                    modalities[modality][date][i]["data"] = data
                
                # 為了避免下游出錯，如果形狀不對，印出形狀供除錯
                print(f"    {subject} {modality} {date} Run {i} - Shape after truncating: {data.shape}")
# ===================================================================
# ===== 插入：數據驗證分析 (Waveform, PSD, Baseline Consistency) =====
# ===================================================================
# 
# 目的：在套用 StandardScaler 之前，驗證數據的基礎品質。
#
if False: # Disabled by user request to focus on DSFE
    print("\n" + "="*50)
    print("開始執行數據驗證分析 (優先度 1, 2)")
    print("="*50)

    # --- 驗證參數 ---
    VALIDATION_LOW_CUT = 1.0   # (Hz) 使用 1Hz 高通濾波器來去除漂移 (處理教授的疑慮)
    VALIDATION_HIGH_CUT = 40.0 # (Hz) 保留 40Hz 以下的腦波頻帶
    VALIDATION_SFREQ = 250
    CH_NAMES = ['C3', 'Cz', 'C4'] # 根據您的程式碼，假設是這 3 個通道

    # --- 參數 (從您的 "第二部分" 複製而來，用於 epoch) ---
    cue_length = 2 * VALIDATION_SFREQ
    mi_length = 4 * VALIDATION_SFREQ
    rest_length = 4 * VALIDATION_SFREQ
    trial_length = cue_length + mi_length + rest_length
    n_trials_expected = 48

    # --- 用於儲存基準線統計數據 ---
    all_baseline_stats = []

    # --- 標記 (修改) ---
    # 我們使用字典來記錄「每個受試者」是否已經畫過圖
    waveform_plot_drawn_per_subject = {}
    psd_plot_drawn_per_subject = {}

    # --- 遍歷所有數據 (在 Part 1 裁切後, Part 2 標準化前) ---
    for subject, modalities in norm_mod_data.items():
        
        # *** 新增 ***
        # 如果這個受試者兩種圖都畫過了，我們可以跳過後續的繪圖檢查
        if subject in waveform_plot_drawn_per_subject and subject in psd_plot_drawn_per_subject:
            # (但我們仍需繼續迴圈，以收集 任務 3 的基準線數據)
            pass 

        for modality in modalities_list:
            if modality not in modalities:
                continue
            for date, datas in modalities[modality].items():
                for i in datas:
                    run_data = datas[i]['data']
                    
                    # 驗證資料形狀
                    if run_data.shape != (3, 120000):
                        print(f"    [驗證] 跳過 {subject} {modality} {date} Run {i} - 預期形狀 (3, 120000), 得到 {run_data.shape}")
                        continue
                    
                    # --- 步驟 1 & 2：執行「驗證用」濾波 ---
                    try:
                        filtered_run_data = ezbandpass_filter(
                            run_data, 
                            VALIDATION_LOW_CUT, 
                            VALIDATION_HIGH_CUT, 
                            fs=VALIDATION_SFREQ, 
                            order=5
                        )
                    except Exception as e:
                        print(f"    [!] 濾波失敗 at {subject} {modality} {date} Run {i}: {e}")
                        continue

                    # --- 任務 1：查看「已濾波」的數據波形 (*** 修改 ***) ---
                    # 檢查這個「受試者」是否還沒畫過波形圖
                    if subject not in waveform_plot_drawn_per_subject:
                        print(f"\n[繪圖] 正在為 Subject: {subject} 繪製波形範例...")
                        plt.figure(figsize=(15, 6))
                        plot_duration_samples = 10 * VALIDATION_SFREQ # 畫 10 秒
                        for ch_idx, ch_name in enumerate(CH_NAMES):
                            plt.subplot(3, 1, ch_idx + 1)
                            plt.plot(np.arange(plot_duration_samples) / VALIDATION_SFREQ, 
                                     filtered_run_data[ch_idx, :plot_duration_samples])
                            plt.title(f"{ch_name} - (已濾波 1-40 Hz)")
                            plt.ylabel("振幅 (µV)")
                        plt.xlabel("時間 (秒)")
                        plt.suptitle(f"任務 1：已濾波波形範例\n(Subject: {subject}, Modality: {modality}, Date: {date}, Run: {i})", fontsize=16)
                        plt.tight_layout()
                        plt.show()
                        # 標記這個受試者已畫過
                        waveform_plot_drawn_per_subject[subject] = True

                    # --- 任務 2：查看 PSD 是否呈現 1/f 特性 (*** 修改 ***) ---
                    # 檢查這個「受試者」是否還沒畫過 PSD 圖
                    if subject not in psd_plot_drawn_per_subject:
                        print(f"\n[繪圖] 正在為 Subject: {subject} 繪製 PSD 範例...")
                        plt.figure(figsize=(10, 5))
                        for ch_idx, ch_name in enumerate(CH_NAMES):
                            # 使用 Welch 方法計算 PSD
                            freqs, psd_values = welch(
                                filtered_run_data[ch_idx, :], 
                                fs=VALIDATION_SFREQ, 
                                nperseg=VALIDATION_SFREQ * 2 # 2 秒的窗格
                            )
                            # 轉換為 dB 單位 (log scale) 更容易觀察 1/f
                            plt.plot(freqs, 10 * np.log10(psd_values), label=f"{ch_name}")
                        
                        plt.xlim(VALIDATION_LOW_CUT, VALIDATION_HIGH_CUT + 10) # 顯示到 50 Hz
                        plt.xlabel("頻率 (Hz)")
                        plt.ylabel("功率譜密度 (dB/Hz)")
                        plt.title(f"任務 2：PSD 1/f 特性檢視 (1-40 Hz 濾波後)\n(Subject: {subject}, Modality: {modality}, Date: {date}, Run: {i})")
                        plt.legend()
                        plt.grid(True)
                        plt.show()
                        # 標記這個受試者已畫過
                        psd_plot_drawn_per_subject[subject] = True

                    # --- 任務 3：基準線一致性與穩定性分析 ---
                    # (這部分程式碼不變，它本來就會處理所有受試者)
                    for trial_idx in range(n_trials_expected):
                        start_index = trial_idx * trial_length
                        end_index = start_index + trial_length
                        if end_index > filtered_run_data.shape[1]:
                            continue
                        
                        trial_data_filtered = filtered_run_data[:, start_index:end_index]
                        
                        rest_start_sample = cue_length + mi_length
                        rest_end_sample = trial_length

                        # 提取「基線期」 (根據您的建議，使用 MI 之前的 Cue 期: 0s-2s)
                        baseline_start_sample = 0
                        baseline_end_sample = cue_length # 這是 2*sfreq (前 2 秒)
                        baseline_data = trial_data_filtered[:, baseline_start_sample:baseline_end_sample]
                        
                        baseline_stds = np.std(baseline_data, axis=1) # 沿著時間軸計算
                        
                        for ch_idx, ch_name in enumerate(CH_NAMES):
                            all_baseline_stats.append({
                                "subject": subject,
                                "modality": modality,
                                "date": date,
                                "run": i,
                                "trial": trial_idx,
                                "channel": ch_name,
                                "baseline_std": baseline_stds[ch_idx]
                            })

    print("\n" + "="*50)
    print("基礎驗證完成，開始繪製基準線一致性報告")
    print("="*50)

    # --- 繪製 任務 3 的結果 ---
    # (這部分程式碼不變，它會繪製所有受試者的總結圖表)
    if not all_baseline_stats:
        print("[!] 沒有收集到任何基準線統計數據，無法繪圖。")
    else:
        stats_df = pd.DataFrame(all_baseline_stats)
        
        # 移除極端異常值 (例如 > 200 µV)，避免圖表失真
        stats_df = stats_df[stats_df['baseline_std'] < 200]
        
        # 圖 3：比較「不同受試者」的基準線穩定性
        plt.figure(figsize=(15, 7))
        sns.boxplot(data=stats_df, x='subject', y='baseline_std', hue='channel')
        plt.title("任務 3a：基準線穩定性 (比較不同受試者)\n(休息期 1-40Hz 濾波後的標準差)")
        plt.ylabel("標準差 (µV)")
        plt.xlabel("受試者")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        # 圖 4：比較「同一受試者，不同日期/實驗」的基準線穩定性
        subjects_to_plot = stats_df['subject'].unique()
        for subj in subjects_to_plot:
            subj_df = stats_df[stats_df['subject'] == subj]
            # 檢查是否有足夠的日期可供比較
            if subj_df['date'].nunique() > 1:
                plt.figure(figsize=(15, 7))
                sns.boxplot(data=subj_df, x='date', y='baseline_std', hue='channel')
                plt.title(f"任務 3b：{subj} 的內部實驗一致性\n(比較不同日期的基準線標準差)")
                plt.ylabel("標準差 (µV)")
                plt.xlabel("實驗日期")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.show()
            else:
                print(f"[!] 受試者 {subj} 只有一個日期的數據，跳過內部一致性比較。")

    print("\n" + "="*50)
    print("驗證分析結束。現在繼續執行您的原始程式碼...")
    print("="*50 + "\n")

# ===== 第二部分：試驗分段與標準化（產生 epoch 結構） =====
# ... (您原有的程式碼從這裡繼續) ...
# ===== 第二部分：試驗分段與標準化（產生 epoch 結構） =====
scaler = StandardScaler()
for subject, modalities in norm_mod_data.items():
    print("Processing subject:", subject)
    for modality in modalities_list:
        if modality not in modalities:
            continue
        for date, data_n in modalities[modality].items():
            print("  date:", date)
            for i in modalities[modality][date]:
                print("    run:", i)
                data = modalities[modality][date][i]["data"]
                # 防呆檢查
                if data.shape != (3, 120000):
                    print(f"      [!] Skipping run {i} due to incorrect shape: {data.shape}. Expected (3, 120000).")
                    continue
                sfreq = 250
                cue_length = 2 * sfreq
                mi_length = 4 * sfreq
                rest_length = 4 * sfreq
                trial_length = cue_length + mi_length + rest_length
                segmented_data = {}
                data_epoch = []
                # 假設每個 run 有 48 個 trials（與原程式一致）
                n_trials_expected = 48
                for trial in range(n_trials_expected):
                    start_index = trial * trial_length
                    end_index = start_index + trial_length
                    if end_index > data.shape[1]:
                        print(f"      [!] 試驗 {trial} 超出資料長度，跳過。")
                        continue
                    trial_data = data[:, start_index:end_index]
                    # 對每個 trial 做標準化 (時間點為樣本是 row 的轉置)
                    transformed_data = scaler.fit_transform(trial_data.T).T
                    segmented_data[f"trial_{trial}"] = {
                        "all": transformed_data,
                        "cue": trial_data[:, :cue_length],
                        "mi": trial_data[:, cue_length:cue_length + mi_length],
                        "rest": trial_data[:, cue_length + mi_length:trial_length]
                    }
                    data_epoch.append(trial_data)
                
                if len(data_epoch) == 0:
                    print(f"      [!] Run {i} 沒有有效的試驗 epoch，跳過更新。")
                    continue
                segmented_data["trial_all"] = data
                segmented_data["epoch"] = np.array(data_epoch)
                
                print("      segmented_data epoch shape:", segmented_data["epoch"].shape)
                norm_mod_data[subject][modality][date][i]["data"] = segmented_data

# ===== Interactive selector (保留) =====
# 在 Jupyter 中使用 ipywidgets 顯示下拉選單
# try:
#     %matplotlib inline  # 若在 IPython / Jupyter 環境下
# except Exception:
#     pass
import ipywidgets as widgets
from IPython.display import display, clear_output

def interactive_selector(norm_mod_data):  
    subjects = list(norm_mod_data.keys())
    subj_dropdown = widgets.SelectMultiple(options=subjects, description='Subjects:')
    modality_dropdown = widgets.SelectMultiple(options=modalities_list, description='Modalities:')
    date_dropdown = widgets.Dropdown(description='Date:')
    run_dropdown = widgets.Dropdown(description='Run:')
    current_subject = None
    current_modality = None

    def update_dates(*args):
        nonlocal current_subject, current_modality
        if len(subj_dropdown.value) > 0:
            current_subject = subj_dropdown.value[0]  # 暫時支援單一，為簡化
            if len(modality_dropdown.value) > 0:
                current_modality = modality_dropdown.value[0]
                dates = list(norm_mod_data[current_subject].get(current_modality, {}).keys())
                date_dropdown.options = dates
                if dates:
                    date_dropdown.value = dates[0]

    def update_runs(*args):
        if current_subject and current_modality and date_dropdown.value:
            runs = list(norm_mod_data[current_subject][current_modality][date_dropdown.value].keys())
            run_dropdown.options = runs
            if runs:
                run_dropdown.value = runs[0]

    subj_dropdown.observe(update_dates, names='value')
    modality_dropdown.observe(update_dates, names='value')
    date_dropdown.observe(update_runs, names='value')
    update_dates()
    update_runs()
    ui = widgets.VBox([modality_dropdown, subj_dropdown, date_dropdown, run_dropdown])  # 修改順序，先模態
    display(ui)
    
    def get_selection():
        selections = {}
        for mod in modality_dropdown.value:
            for subj in subj_dropdown.value:
                if mod not in selections:
                    selections[mod] = {}
                selections[mod][subj] = {'date': date_dropdown.value, 'run': run_dropdown.value}
        print(f"Selected: {selections}")
        return selections
    return get_selection

# 注意：Jupyter部分簡化，生產中可擴展多選

# ===== 時間/事件參數 =====
cue_start = 0
mi_start = 2
baseline_start = 0
baseline_end = 2
trial_len = 10

# ===== 修改後的 event_id（2 類）=====
# lift -> 0, fist -> 1
event_id = {
    'lift': 0,
    'fist': 1
}

# ===== CLI / Session 選擇工具（修改為菜單式，先選擇模態） =====
def get_subject_sessions(norm_mod_data, subject, modality):
    rows = []
    if modality in norm_mod_data[subject]:
        for date, runs in norm_mod_data[subject][modality].items():
            for run in runs:
                rows.append({'subject': subject, 'modality': modality, 'date': date, 'run': run})
    df = pd.DataFrame(rows)
    return df

def cli_select_subjects_modalities_sessions(norm_mod_data):
    selected = {}
    
    # 先選擇模態 (菜單式)
    while True:
        print("\n=== 模態選擇菜單 ===")
        for idx, mod in enumerate(modalities_list, 1):
            print(f"{idx}: {mod}")
        print("0: 完成模態選擇並繼續")
        choice = input("請輸入選擇 (多選用逗號分隔，如 1,2): ")
        if choice == '0':
            break
        try:
            selected_mods_idx = [int(idx.strip()) for idx in choice.split(",") if idx.strip()]
            selected_mods = [modalities_list[i-1] for i in selected_mods_idx if 1 <= i <= len(modalities_list)]
        except ValueError:
            print("無效輸入，請重試。")
            continue
        if not selected_mods:
            print("無有效選擇，請重試。")
            continue
        break  # 假設單次選擇，否則可loop添加
    
    if not selected_mods:
        selected_mods = modalities_list  # 默认全部
    
    # 選擇受試者 (過濾有選定模態的)
    subjects = list(norm_mod_data.keys())
    avail_subjects = [subj for subj in subjects if any(mod in norm_mod_data[subj] for mod in selected_mods)]
    while True:
        print("\n=== 受試者選擇菜單 (僅顯示有選定模態的) ===")
        for idx, subj in enumerate(avail_subjects, 1):
            avail_mods = [mod for mod in selected_mods if mod in norm_mod_data[subj]]
            print(f"{idx}: {subj} (Modalities: {', '.join(avail_mods)})")
        print("0: 完成受試者選擇並繼續")
        choice = input("請輸入選擇 (多選用逗號分隔，如 1,3): ")
        if choice == '0':
            break
        try:
            selected_subjs_idx = [int(idx.strip()) for idx in choice.split(",") if idx.strip()]
            selected_subjects = [avail_subjects[i-1] for i in selected_subjs_idx if 1 <= i <= len(avail_subjects)]
        except ValueError:
            print("無效輸入，請重試。")
            continue
        if not selected_subjects:
            print("無有效選擇，請重試。")
            continue
        break
    
    if not selected_subjects:
        selected_subjects = avail_subjects  # 默认全部
    
    for subject in selected_subjects:
        selected[subject] = {}
        for modality in selected_mods:
            if modality not in norm_mod_data[subject]:
                continue
            print(f"\n=== {subject} - {modality} session 選擇菜單 ===")
            df = get_subject_sessions(norm_mod_data, subject, modality)
            if df.empty:
                print("此模態沒有資料")
                continue
            print(df)
            # include/exclude 邏輯，同原
            include = input("請輸入要保留的 index（逗號分隔，如 0,2,5），全部都要就直接 Enter: ")
            if include.strip() == "":
                include_idx = list(range(len(df)))
            else:
                include_idx = [int(idx.strip()) for idx in include.split(",") if idx.strip() != '']
            df = df.iloc[include_idx].reset_index(drop=True)
            print("保留的 session：")
            print(df)
            if df.empty:
                continue
            
            exclude = input("請輸入不要的 index（逗號分隔，如 0,2,5），全部都要就直接 Enter: ")
            if exclude.strip() == "":
                exclude_idx = []
            else:
                exclude_idx = [int(idx.strip()) for idx in exclude.split(",") if idx.strip() != '']
            filtered_df = df.drop(exclude_idx).reset_index(drop=True)
            print("最終保留的 session：")
            print(filtered_df)
            selected[subject][modality] = filtered_df
    return selected

def get_session_data(norm_mod_data, subject, modality, filtered_df):
    sessions = []
    for _, row in filtered_df.iterrows():
        date = row['date']
        run = row['run']
        data = norm_mod_data[subject][modality][date][run]['data']
        label = norm_mod_data[subject][modality][date][run]['label']
        sessions.append({
            'subject': subject,
            'modality': modality,
            'date': date,
            'run': run,
            'data': data,
            'label': label
        })
    return sessions


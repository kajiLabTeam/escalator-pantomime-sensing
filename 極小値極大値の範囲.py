import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import numpy as np
import japanize_matplotlib

# バターワースローパスフィルタ関数の定義
def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# ジャイロスコープデータを読み込む
gyroscope_data = pd.read_csv('Gyroscope4,4.csv')

# ジャイロスコープのX軸データ（角速度）を取得
gyro_x = gyroscope_data['Gyroscope x (rad/s)']

# サンプリング周波数を設定
time_gyro = gyroscope_data['Time (s)'].to_numpy()  # NumPy配列に変換
fs_gyro = 1 / (time_gyro[1] - time_gyro[0])  # ジャイロスコープデータのサンプリング周波数

# ローパスフィルタを適用してデータを平滑化
cutoff_freq = 3  # カットオフ周波数
filtered_gyro_x = butter_lowpass_filter(gyro_x, cutoff_freq, fs_gyro)

# 角速度を積分して角度を計算
angles_rad = np.cumsum(filtered_gyro_x) * (time_gyro[1] - time_gyro[0])  # 各時間間隔での積分

# ラジアンから度数に変換
angles_deg = angles_rad * (180 / np.pi)

# 横軸（時間）範囲の指定
start_time =15.03405807092625  # 開始時間（秒）
end_time = 25.3140836351472    # 終了時間（秒）

# 指定範囲のデータを取得
time_mask = (time_gyro >= start_time) & (time_gyro <= end_time)
time_range = time_gyro[time_mask]  # NumPy配列のまま使用
angles_range = angles_deg[time_mask]  # NumPy配列のまま使用

# 指定範囲内のピークと極小値を検出
angle_peaks_range, _ = find_peaks(angles_range, height=20)
angle_troughs_range, _ = find_peaks(-angles_range)

# # ピーク地点の平均を計算han
# if len(angle_peaks_range) > 0:
#     average_peak_angle_range = np.mean(angles_range[angle_peaks_range])
#     print(f"指定範囲内のピーク地点の平均角度: {average_peak_angle_range:.2f} 度")
# else:
#     print("指定範囲内に30度以上のピークは検出されませんでした。")

# 極大値と極小値の差を求める
differences = []
for i in range(min(len(angle_peaks_range), len(angle_troughs_range))):
    peak_index = angle_peaks_range[i]
    trough_index = angle_troughs_range[i]
    
    # 差を計算
    difference = angles_range[peak_index] - angles_range[trough_index]
    differences.append(difference)

# 分散と標準偏差、平均を計算
if differences:
    variance = np.var(differences)
    std_dev = np.std(differences)
    mean_diff = np.mean(differences)
    
    # 結果を表示
    for i, diff in enumerate(differences):
        print(f"極大値{i + 1} と 極小値{i + 1} の差: {diff:.2f} 度")
    
    print(f"\n差の平均: {mean_diff:.2f} 度")
    print(f"差の分散: {variance:.2f}")
    print(f"差の標準偏差: {std_dev:.2f}")
else:
    print("差の計算に使用できるデータがありません。")

# 極小値の平均、分散、標準偏差を計算
if len(angle_troughs_range) > 0:
    trough_values = angles_range[angle_troughs_range]
    trough_mean = np.mean(trough_values)
    trough_variance = np.var(trough_values)
    trough_std_dev = np.std(trough_values)
    
    print(f"\n極小値の平均: {trough_mean:.2f} 度")
    print(f"極小値の分散: {trough_variance:.2f}")
    print(f"極小値の標準偏差: {trough_std_dev:.2f}")
else:
    print("極小値が検出されませんでした。")

# 極大値の平均、分散、標準偏差を計算
if len(angle_peaks_range) > 0:
    peak_values = angles_range[angle_peaks_range]
    peak_mean = np.mean(peak_values)
    peak_variance = np.var(peak_values)
    peak_std_dev = np.std(peak_values)

    print(f"\n極大値の平均: {peak_mean:.2f} 度")
    print(f"極大値の分散: {peak_variance:.2f}")
    print(f"極大値の標準偏差: {peak_std_dev:.2f}")
else:
    print("極大値が検出されませんでした。")

# 全体のプロットと範囲内のピーク/極小値のプロット
plt.figure(figsize=(10, 6))

# 全体の角度データをプロット
plt.plot(time_gyro, angles_deg)

# 指定範囲内のピークと極小値をプロット
plt.plot(time_range[angle_peaks_range], angles_range[angle_peaks_range], "o", color='red')
plt.plot(time_range[angle_troughs_range], angles_range[angle_troughs_range], "o", color='blue')

# 番号を付けてプロット
for i, peak in enumerate(angle_peaks_range):
    plt.text(time_range[peak], angles_range[peak], str(i + 1), color='red', fontsize=20)

for i, trough in enumerate(angle_troughs_range):
    plt.text(time_range[trough], angles_range[trough], str(i + 1), color='blue', fontsize=20)

# グラフの設定
# plt.title('角速度からの角度の積分 (度数) ',fontsize=20)
plt.xlabel('時間 (秒)',fontsize=25)
plt.ylabel('角度 (度)',fontsize=25)
plt.legend()
plt.grid(True)
# メモリの大きさを変更
plt.tick_params(axis='both', labelsize=25)  # メモリのラベルサイズを15に設定

plt.show()

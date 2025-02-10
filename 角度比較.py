import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
import japanize_matplotlib

# バターワースローパスフィルタ関数の定義
def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# CSVファイルを読み込む
gyro_data_left = pd.read_csv('Gyroscope24_2.csv')
gyro_data_right = pd.read_csv('Gyroscope8_2.csv')

# ジャイロスコープのX軸データ（角速度）を取得
gyro_x_left = gyro_data_left['Gyroscope x (rad/s)']
gyro_x_right = gyro_data_right['Gyroscope x (rad/s)']

# サンプリング周波数を設定
time_left = gyro_data_left['Time (s)']
time_right = gyro_data_right['Time (s)']
fs_left = 1 / (time_left[1] - time_left[0])
fs_right = 1 / (time_right[1] - time_right[0])

# ローパスフィルタを適用してデータを平滑化
cutoff_freq = 3  # カットオフ周波数
filtered_gyro_x_left = butter_lowpass_filter(gyro_x_left, cutoff_freq, fs_left)
filtered_gyro_x_right = butter_lowpass_filter(gyro_x_right, cutoff_freq, fs_right)

# 積分結果を計算
angles_rad_left = np.cumsum(filtered_gyro_x_left) * (time_left[1] - time_left[0])
angles_rad_right = np.cumsum(filtered_gyro_x_right) * (time_right[1] - time_right[0])

# ラジアンから度数に変換
angles_deg_left = angles_rad_left * (180 / np.pi)
angles_deg_right = angles_rad_right * (180 / np.pi)

# プロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# 左側のプロット
ax1.plot(time_left, angles_deg_left)
ax1.set_title('階段', fontsize=30)  # タイトルの文字サイズ
ax1.set_xlabel('時間 (秒)', fontsize=20)  # 横軸の文字サイズ
ax1.set_ylabel('角度 (度)', fontsize=20)  # 縦軸の文字サイズ
ax1.tick_params(axis='both', labelsize=20)  # メモリの文字サイズ
ax1.grid(True)
ax1.legend()

# 右側のプロット
ax2.plot(time_right, angles_deg_right)
ax2.set_title('パントマイム', fontsize=30)  # タイトルの文字サイズ
ax2.set_xlabel('時間 (秒)', fontsize=20)  # 横軸の文字サイズ
ax2.tick_params(axis='both', labelsize=20)  # メモリの文字サイズ
ax2.grid(True)
ax2.legend()

# プロットを表示
plt.tight_layout()
plt.show()
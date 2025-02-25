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
gyroscope_data = pd.read_csv('Gyroscope8_2.csv')

# ジャイロスコープのX軸データ（角速度）を取得
gyro_x = gyroscope_data['Gyroscope y (rad/s)']

# サンプリング周波数を設定
time_gyro = gyroscope_data['Time (s)']
fs_gyro = 1 / (time_gyro[1] - time_gyro[0])  # ジャイロスコープデータのサンプリング周波数

# ローパスフィルタを適用してデータを平滑化
cutoff_freq = 3  # カットオフ周波数
filtered_gyro_x = butter_lowpass_filter(gyro_x, cutoff_freq, fs_gyro)

# 角速度を積分して角度を計算する関数
def integrate_angular_velocity(gyro_data, time_data):
    # np.trapzを使って数値積分を行う
    angle = np.trapz(gyro_data, time_data)
    return angle

# フィルタリングされた角速度データを積分
angles_rad = np.cumsum(filtered_gyro_x) * (time_gyro[1] - time_gyro[0])  # 各時間間隔での積分

# ラジアンから度数に変換
angles_deg = angles_rad * (180 / np.pi)

# 積分結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(time_gyro, angles_deg, label='Integrated Angle (degrees)')
plt.title('角速度からの角度の積分 (度数)')
plt.xlabel('時間 (秒)')
plt.ylabel('角度 (度)')
plt.legend()
plt.grid(True)
plt.show()

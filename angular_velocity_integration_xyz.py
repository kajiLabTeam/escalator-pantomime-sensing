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

# ジャイロスコープデータを読み込む
gyroscope_data = pd.read_csv('Gyroscope24_2.csv')

# ジャイロスコープの各軸データ（角速度）を取得
gyro_x = gyroscope_data['Gyroscope x (rad/s)']
gyro_y = gyroscope_data['Gyroscope y (rad/s)']
gyro_z = gyroscope_data['Gyroscope z (rad/s)']

# サンプリング周波数を設定
time_gyro = gyroscope_data['Time (s)']
fs_gyro = 1 / (time_gyro[1] - time_gyro[0])  # ジャイロスコープデータのサンプリング周波数

# ローパスフィルタを適用してデータを平滑化
cutoff_freq = 3  # カットオフ周波数
filtered_gyro_x = butter_lowpass_filter(gyro_x, cutoff_freq, fs_gyro)
filtered_gyro_y = butter_lowpass_filter(gyro_y, cutoff_freq, fs_gyro)
filtered_gyro_z = butter_lowpass_filter(gyro_z, cutoff_freq, fs_gyro)

# 角速度を積分して角度を計算する関数
def integrate_angular_velocity(gyro_data, time_data):
    return np.cumsum(gyro_data) * (time_data[1] - time_data[0])  # 各時間間隔での積分

# フィルタリングされた角速度データを積分
angles_rad_x = integrate_angular_velocity(filtered_gyro_x, time_gyro)
angles_rad_y = integrate_angular_velocity(filtered_gyro_y, time_gyro)
angles_rad_z = integrate_angular_velocity(filtered_gyro_z, time_gyro)

# ラジアンから度数に変換
angles_deg_x = angles_rad_x * (180 / np.pi)
angles_deg_y = angles_rad_y * (180 / np.pi)
angles_deg_z = angles_rad_z * (180 / np.pi)

# 積分結果をプロット
plt.figure(figsize=(12, 8))
plt.plot(time_gyro, angles_deg_x, label='X軸角度 (度)', color='r')
plt.plot(time_gyro, angles_deg_y, label='Y軸角度 (度)', color='g')
plt.plot(time_gyro, angles_deg_z, label='Z軸角度 (度)', color='b')
plt.title('角速度からの角度の積分 (度数)')
plt.xlabel('時間 (秒)')
plt.ylabel('角度 (度)')
plt.legend()
plt.grid(True)
plt.show()

# 積分結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(time_gyro, angles_deg_x, label='Integrated Angle (degrees)')
plt.title('角速度からの角度の積分 (度数)')
plt.xlabel('時間 (秒)')
plt.ylabel('角度 (度)')
plt.legend()
plt.grid(True)
plt.show()

# 積分結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(time_gyro, angles_deg_y, label='Integrated Angle (degrees)')
plt.title('角速度からの角度の積分 (度数)')
plt.xlabel('時間 (秒)')
plt.ylabel('角度 (度)')
plt.legend()
plt.grid(True)
plt.show()

# 積分結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(time_gyro, angles_deg_z, label='Integrated Angle (degrees)')
plt.title('角速度からの角度の積分 (度数)')
plt.xlabel('時間 (秒)')
plt.ylabel('角度 (度)')
plt.legend()
plt.grid(True)
plt.show()
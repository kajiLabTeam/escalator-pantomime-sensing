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
gyroscope_data = pd.read_csv('Gyroscope12_2.csv')

# ジャイロスコープのX軸データ（角速度）を取得
gyro_x = gyroscope_data['Gyroscope x (rad/s)']

# サンプリング周波数を設定
time_gyro = gyroscope_data['Time (s)']
fs_gyro = 1 / (time_gyro[1] - time_gyro[0])  # ジャイロスコープデータのサンプリング周波数

# ローパスフィルタを適用してデータを平滑化
cutoff_freq = 3  # カットオフ周波数
filtered_gyro_x = butter_lowpass_filter(gyro_x, cutoff_freq, fs_gyro)


# 角速度を積分して角度を計算
angles_rad = np.cumsum(filtered_gyro_x) * (time_gyro[1] - time_gyro[0])  # 各時間間隔での積分

# ラジアンから度数に変換
angles_deg = angles_rad * (180 / np.pi)

# 角度データのピークを検出（70度以上）
angle_peaks, _ = find_peaks(angles_deg, height=30)

# ピーク地点の平均を計算
if len(angle_peaks) > 0:
    average_peak_angle = np.mean(angles_deg[angle_peaks])
    print(f"ピーク地点の平均角度: {average_peak_angle:.2f} 度")
else:
    print("70度以上のピークは検出されませんでした。")

# 積分結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(time_gyro, angles_deg, label='Integrated Angle (degrees)')
plt.plot(time_gyro[angle_peaks], angles_deg[angle_peaks], "x",  color='red')
plt.title('角速度からの角度の積分 (度数)')
plt.xlabel('時間 (秒)')
plt.ylabel('角度 (度)')
plt.legend()
plt.grid(True)
plt.show()
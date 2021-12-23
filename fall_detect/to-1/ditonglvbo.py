from scipy import signal
def low_pass_filter(data):
    b, a = signal.butter(8, 0.4, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  # data为要过滤的信号
    return filtedData
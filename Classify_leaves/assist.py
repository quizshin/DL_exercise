import time
import numpy as np

# # 定义累加器的类，用于累加每个batch的运行状态数据（损失和准确度）
class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):  # 初始化，n为累加器的列数
        self.data = [0.0] * n  # list * int 意思是将数组重复 int 次并依次连接形成一个新数组

    def add(self, *args):  # data和args对应列累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):  # 重置
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):  # 索引
        return self.data[idx]


# # 定义记录多次运行时间的Timer类
class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """初始化"""
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并在列表中记录时间"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间的总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
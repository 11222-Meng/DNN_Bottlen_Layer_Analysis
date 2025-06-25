import psutil
import os
import time


def monitor_memory_usage(interval=0.01, stop_event=None):
    """
    监控当前进程的内存占用情况。
    :param interval: 监控间隔时间（秒）
    :param stop_event: 停止事件，用于控制监控线程退出
    :return: 内存占用列表（单位：MB）
    """
    process = psutil.Process(os.getpid())  # 获取当前进程
    memory_usage = []
    while not stop_event.is_set():  # 检查停止事件
        try:
            # 获取当前内存占用并转换为MB
            mem = process.memory_info().rss / 1024 / 1024
            memory_usage.append(mem)
            time.sleep(interval)  # 等待指定间隔
        except KeyboardInterrupt:
            break  # 用户中断时退出

    # 计算峰值内存
    if memory_usage:
        print(f"\nPeak memory usage: {max(memory_usage):.2f} MB")
    return memory_usage
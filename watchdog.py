#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
看门狗代码，每日重启交易系统

此代码会启动交易系统（假设文件名为 trading.py）作为子进程，
并每天在指定的重启时间（默认凌晨 0 点）终止后重新启动交易系统。
"""

import subprocess
import time
import datetime
import sys
import signal


def start_trading():
    """
    启动交易系统的子进程。
    此处假设交易代码位于 trading.py，你可以替换为实际的文件路径或命令。
    """
    try:
        process = subprocess.Popen([sys.executable, "run.py"])
        print(f"Started trading process with PID: {process.pid}")
        return process
    except Exception as e:
        print("启动交易系统失败:", e)
        sys.exit(1)


def get_seconds_until_restart(target_hour=0, target_minute=0):
    """
    计算距离下一次重启（默认为每天凌晨0点）的秒数
    """
    now = datetime.datetime.now()
    # 构造今天的重启目标时间
    target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target_time <= now:
        # 如果当前时间已过目标时间，则目标时间设为明天
        target_time += datetime.timedelta(days=1)
    return (target_time - now).total_seconds()


def restart_process(process):
    """
    优雅终止现有的交易系统进程，如果无法正常退出则强制杀死进程
    """
    try:
        print("尝试优雅终止交易系统进程...")
        process.terminate()  # 发送终止信号
        process.wait(timeout=30)  # 等待最多 30 秒
        print("交易系统进程已优雅退出。")
    except Exception as e:
        print("优雅结束失败，使用 kill 终止进程。", e)
        process.kill()


def main():
    trading_process = start_trading()

    try:
        while True:
            # 计算距离下一个重启时间（默认为每天凌晨 0 点）的秒数
            sleep_duration = get_seconds_until_restart(target_hour=0, target_minute=0)
            print(f"看门狗将在 {sleep_duration:.0f} 秒后重启交易系统。")

            # 睡眠直到预定的重启时间
            time.sleep(sleep_duration)

            print("到达重启时间，准备重启交易系统...")
            restart_process(trading_process)
            # 在重启前等待几秒钟，保证进程完全退出
            time.sleep(5)
            trading_process = start_trading()
    except KeyboardInterrupt:
        print("看门狗收到退出信号，正在终止交易系统...")
        restart_process(trading_process)
        sys.exit(0)
    except Exception as e:
        print("看门狗发生异常:", e)
        restart_process(trading_process)
        sys.exit(1)


if __name__ == "__main__":
    main()
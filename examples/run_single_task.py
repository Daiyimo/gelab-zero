import os
import sys
import time
import random

if "." not in sys.path:
    sys.path.append(".")

from copilot_agent_client.pu_client import evaluate_task_on_device
from copilot_front_end.mobile_action_helper import list_devices, get_device_wm_size
from copilot_agent_server.local_server import LocalServer

tmp_server_config = {
    "log_dir": "running_log/server_log/os-copilot-local-eval-logs/traces",
    "image_dir": "running_log/server_log/os-copilot-local-eval-logs/images",
    "debug": False
}

local_model_config = {
    "task_type": "parser_0922_summary",
    "model_config": {
        "model_name": "gelab-zero-4b-preview",
        "model_provider": "local",
        "args": {
            "temperature": 0.1,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "max_tokens": 4096,
        },
        # optional to resize image
        # "resize_config": {
        #     "is_resize": True,
        #     "target_image_size": (756, 756)
        # }
    },
    "max_steps": 400,
    "delay_after_capture": 2,
    "debug": False,
    "stream": True,
}

# ===== 全局变量：用于记录每步耗时和Token消耗 =====
_step_times = []
_total_input_tokens = 0
_total_output_tokens = 0

# ===== 新增：Token 模拟函数 =====
def simulate_token_usage():
    # 假设输入 Token (屏幕截图、DOM、历史等) 范围
    input_tokens = random.randint(1500, 2000) 
    # 假设输出 Token (生成的动作、解释文本) 范围
    output_tokens = random.randint(50, 150)
    return input_tokens, output_tokens

# ===== 核心修改：包装 automate_step 方法，仅处理耗时和Token统计 =====
def wrap_automate_step_with_timing(server_instance):
    original_method = server_instance.automate_step

    def timed_automate_step(payload):
        global _step_times, _total_input_tokens, _total_output_tokens  # 👈 关键修复：声明全局变量
        step_start = time.time()
        result = {}
        try:
            # 执行原始的自动化步骤
            result = original_method(payload)
            # 【核心修改】不从 result 中提取，而是调用模拟函数获取 Token 
            input_tokens, output_tokens = simulate_token_usage()
            # 累加总 Token
            _total_input_tokens += input_tokens
            _total_output_tokens += output_tokens
        except Exception as e:
            # 打印导致错误的具体信息，然后重新抛出异常
            print(f"An error occurred during step automation: {e}")
            raise
        finally:
            duration = time.time() - step_start
            _step_times.append(duration)
            # 仅打印每步耗时，不打印 Token 信息
            print(f"Step {len(_step_times)} took: {duration:.2f} seconds")
        return result

    # 替换实例方法
    server_instance.automate_step = timed_automate_step


if __name__ == "__main__":
    # The device ID you want to use
    device_id = list_devices()[0]
    device_wm_size = get_device_wm_size(device_id)
    device_info = {
        "device_id": device_id,
        "device_wm_size": device_wm_size
    }

    task = "在微信里搜索数字生命卡兹克账号，并关注账号"
    tmp_rollout_config = local_model_config
    l2_server = LocalServer(tmp_server_config)
    
    # 注入计时和Token统计逻辑
    wrap_automate_step_with_timing(l2_server)

    # 执行任务并计总时间
    total_start = time.time()
    # Disable auto reply
    evaluate_task_on_device(l2_server, device_info, task, tmp_rollout_config, reflush_app=True)
    total_time = time.time() - total_start

    # 在最后打印总结（保留总的 Token 消耗）
    print("\n" + "="*50)
    print(f" 任务执行完毕！")
    print("-" * 20)
    print(f"总计执行时间为 {total_time:.2f} 秒")
    print(f"总步数: {len(_step_times)}")
    print("-" * 20)
    print(f"总计输入 Token (模拟): {_total_input_tokens}")
    print(f"总计输出 Token (模拟): {_total_output_tokens}")
    print(f"总 Token 消耗 (模拟): {_total_input_tokens + _total_output_tokens}")
    print("="*50)

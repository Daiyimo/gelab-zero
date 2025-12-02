import os
import sys
if "." not in sys.path:
    sys.path.append(".")

from copilot_agent_client.pu_client import evaluate_task_on_device

from copilot_front_end.mobile_action_helper import list_devices

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
        }
    },

    "max_steps": 400,
    "delay_after_capture": 2,
    "debug": False
}

if __name__ == "__main__":

    # The device ID you want to use
    device_id = list_devices()[0]
    

    from copilot_front_end.mobile_action_helper import get_device_wm_size

    device_wm_size = get_device_wm_size(device_id)
    device_info = {
        "device_id": device_id,
        "device_wm_size": device_wm_size
    }

    from copilot_agent_server.local_server import LocalServer

    # task = "打开微信，给柏茗，发helloworld"
    # task = "打开 给到 app，在主页，下滑寻找，员工权益-奋斗食代，帮我领劵。如果不能领取就退出。"
    task = "open wechat to send a message 'helloworld' to 'TKJ'"

    tmp_rollout_config = local_model_config

    l2_server = LocalServer(tmp_server_config)

    # Disable auto reply
    evaluate_task_on_device(l2_server, device_info, task, tmp_rollout_config, reflush_app=True)

    # Enable auto reply
    # evaluate_task_on_device(l2_server, device_info, task, tmp_rollout_config, reflush_app=True, auto_reply=True)



    pass
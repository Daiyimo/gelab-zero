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
        # "model_name": "gelab-zero-4b-preview",
        # "model_provider": "local",
        "model_name": "step-gui",
        "model_provider": "stepfun",
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
    "delay_after_capture": 5,
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
        global _step_times, _total_input_tokens, _total_output_tokens
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

    # task = "在微信里搜索数字生命卡兹克账号，并关注账号"
    # task = "在腾讯地图里搜索前往腾讯滨江大厦路线，选择骑行方式开始导航"
    # task = "帮我在应用宝里下载QQ-帮我安装QQ-打开QQ应用"
    # task = "在应用宝商店里分别下载并安装微信读书、腾讯视频、QQ音乐三个应用"
    # task = "使用账号1079546708，密码tUBULi77a@登录QQ"
    # task = "打开微信，用账号wxid_zxtof8j8wjbj12，密码agent@wt进行登录"
    # task = "在“人民日报”的微信公众号里找到“苏炳添宣布退役‘的新闻，并进入这篇文章点赞和收藏"
    # task = "在腾讯体育里切换到“赛程”模块，找到“12月18日”的赛程信息"
    # task = "在腾讯地图里搜索前往上海腾讯滨江大厦路线，选择驾车方式开始导航"
    # task = "在元宝里选择'联网搜索' ，选择'深度思考'发送新对话'整理最近一周黄金etf的行情'"
    # task = "在腾讯新闻里搜索'豆包上线agent'的新闻，筛选资讯，点击查看前三条资讯"
    # task = "在腾讯新闻搜索'豆包上线agent',并总结前三条纯文本资讯的内容"
    # task = "在腾讯新闻里搜索“豆包上线agent”的新闻，并帮我总结一下关键信息"
    # task = "登录理财通，在基金模块搜索“022365”代码，并点击“去看看”进入到这个基金的详情页面"
    # task = "登录腾讯会议，预约一个2025年12月20日 早上8点，会议时长1小时的会议。注意：时间选择器中，向上滑动会让时间变大，向下滑动会让时间变小，请根据当前时间和目标时间8点的差值选择正确的滑动方向"
    # task = "打开腾讯文档，新建excel文档，将新文档命名为“收入统计”，输入列名“客户名称”“收入”"
    # task = "在微信中搜索并打开滴滴小程序"
    # task = "在QQ里的“动态”功能里找到“小游戏”，并切换至榜单页面，告诉我榜单排名第一的小游戏是什么"
    # task = "在微信读书里创建一个名称为《2025》的书单，并搜索《小而美》《商业至简》书籍，将这两本书籍书成功添加到新建的《2025》书单"
    # task = "请在微信读书中按顺序执行以下操作：首先搜索书籍《小而美》，选中该书并创建一个名为《2025》的新书单将它加入；书单创建成功后，再搜索另一本《商业至简》，将其也添加到刚才新建的《2025》书单中"
    # task = "腾讯视频里切换到电影tab下，找到疯狂动物城，发布一条“疯狂动物城也太好看了”的观剧感受"
    # task = "打开腾讯文档，新建excel文档，将新文档命名为“收入统计”，输入列名“客户名称”“收入”"
    task = "打开腾讯文档新建一个 Excel 表格。在表格的第一行第一列输入‘客户名称’，在第一行第二列输入‘收入’。保存后，将该文件重命名为‘收入统计’，最后再在主页查看是否创建成功"



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

import sys

if "." not in sys.path:
    sys.path.append(".")

from megfile import smart_open
import base64
import openai
import yaml
import json
import time

# ===== 新增：全局 step 计数器 =====
# 用于自动追踪函数调用次数（当外部未传入 step 时使用）
_GLOBAL_STEP_COUNTER = 0
_GLOBAL_STEP_LOCK = None  # 用于线程安全（如果需要）

def ask_llm_anything(
    model_provider,
    model_name,
    messages,
    args=None,
    resize_config=None,
    step=None
):
    global _GLOBAL_STEP_COUNTER
    
    # ===== 自动 step 逻辑 =====
    # 如果外部未传入 step，使用全局计数器（自动递增）
    if step is None:
        # 线程安全处理（简单场景可省略锁）
        try:
            _GLOBAL_STEP_COUNTER += 1
            step = _GLOBAL_STEP_COUNTER
        except:
            # 如果多线程环境，使用锁（简单实现）
            import threading
            global _GLOBAL_STEP_LOCK
            if _GLOBAL_STEP_LOCK is None:
                _GLOBAL_STEP_LOCK = threading.Lock()
            with _GLOBAL_STEP_LOCK:
                _GLOBAL_STEP_COUNTER += 1
                step = _GLOBAL_STEP_COUNTER
    
    if args is None:
        args = {
            "max_tokens": 256,
            "temperature": 0.5,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
        }

    with smart_open("model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    if model_provider in model_config:
        openai.api_base = model_config[model_provider]["api_base"]
        openai.api_key = model_config[model_provider]["api_key"]
    else:
        raise ValueError(f"Unknown model provider: {model_provider}")

    # preprocess
    def preprocess_messages(messages):
        for msg in messages:
            if type(msg['content']) == str:
                continue
            assert type(msg['content']) == list
            for content in msg['content']:
                if content['type'] == "text":
                    continue
                assert content['type'] == "image_url" or content['type'] == "image_b64"
                if content['type'] == "image_url":
                    url = content['image_url']['url']
                    # to check if the image is already in base64 format
                    if url.startswith("data:image/"):
                        continue
                    else:
                        image_bytes = smart_open(url, mode="rb").read()
                        b64 = base64.b64encode(image_bytes).decode('utf-8')
                        # to judge the image format
                        if image_bytes[0:4] == b"\x89PNG":
                            content['image_url']['url'] = "data:image/png;base64," + b64
                        elif image_bytes[0:2] == b"\xff\xd8":
                            content['image_url']['url'] = "data:image/jpeg;base64," + b64
                        else:
                            content['image_url']['url'] = "data:image/png;base64," + b64
                else:
                    assert content['type'] == "image_b64"
                    b64 = content['image_b64']['b64_json']
                    del content['image_b64']
                    content['image_url'] = {"url": "data:image/png;base64," + b64}
                    content['type'] = "image_url"

                if resize_config is not None and resize_config.get("is_resize", False) == True:
                    image_url = content['image_url']['url']
                    image_b64_url = image_url.split(",", 1)[1]
                    image_data = base64.b64decode(image_b64_url)
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    image = image.resize(size=resize_config['target_image_size'])
                    image_data = io.BytesIO()
                    image = image.convert('RGB')
                    image.save(image_data, format="JPEG", quality=85)
                    image_data = image_data.getvalue()
                    b64_image = base64.b64encode(image_data).decode('utf-8')
                    content['image_url']['url'] = f"data:image/jpeg;base64,{b64_image}"
        return messages

    messages = preprocess_messages(messages)

    start_time = time.time()
    completion = openai.ChatCompletion.create(
        api_key=openai.api_key,
        api_base=openai.api_base,
        model=model_name,
        messages=messages,
        temperature=args.get("temperature", 0.5),
        top_p=args.get("top_p", 1.0),
        frequency_penalty=args.get("frequency_penalty", 0.0),
        max_tokens=args.get("max_tokens", 100),
    )
    end_time = time.time()
    print(f"LLM {model_name} inference time: {end_time - start_time:.2f} seconds")

    result = completion.choices[0].message['content']
    reasoning = completion.choices[0].message.get("reasoning_content", "")
    if reasoning is not None and len(reasoning) > 0:
        result = "THINK>" + reasoning + "</THINK>" + "\n" + result

    # ===== 美化：分行显示字段 =====
    pretty_result = result
    for keyword in ["explain:", "action:", "point:", "value:", "summary:"]:
        pretty_result = pretty_result.replace(keyword, "\n" + keyword)
    pretty_result = pretty_result.lstrip('\n')

    import re
    pretty_result = re.sub(r'</THINK>\s*\n\s*explain:', '</THINK>\nexplain:', pretty_result)
    # 可选：合并 <THINK> 内部的换行（让 THINK 内容单行显示，更整洁）
    pretty_result = re.sub(r'(<THINK>)(.*?)(</THINK>)', lambda m: m.group(1) + m.group(2).replace('\n', ' ') + m.group(3), pretty_result, flags=re.DOTALL)

    # ===== 修改日志头尾：使用自动 step 编号 =====
    start_line = f"-------------- Step {step} start --------------"
    end_line = f"-------------- Step {step} end --------------"

    print(f"LLM {model_name} says:\n{start_line}\n{pretty_result}\n{end_line}")

    return result

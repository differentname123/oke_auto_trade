import asyncio
import fastapi_poe as fp
import os

from common_utils import get_config

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
async def chat(api_key, bot_name):
    messages = []  # 保存对话历史
    while True:
        user_input = input("你：")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break
        messages.append(fp.ProtocolMessage(role="user", content=user_input))
        response_text = ""
        async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=api_key):
            if partial.text:
                response_text += partial.text
        print(f"{bot_name}：{response_text}")
        # 将 role 从 "assistant" 修改为 "bot"
        messages.append(fp.ProtocolMessage(role="bot", content=response_text))

api_key = get_config('poe_api_key')
bot_name = "GPT-3.5-Turbo"

asyncio.run(chat(api_key, bot_name))
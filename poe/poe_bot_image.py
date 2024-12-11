import asyncio
import re

import aiohttp
import fastapi_poe as fp
from fastapi_poe.types import PartialResponse, Attachment
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
        if response_text:
            print(f"{bot_name}：{response_text}")
            # 从响应文本中提取图像 URL 并下载
            await extract_and_save_image(response_text)
        # 将机器人的回复添加到对话历史中
        messages.append(fp.ProtocolMessage(role="bot", content=response_text))

async def extract_and_save_image(text):
    # 使用正则表达式提取 Markdown 格式的图像链接 ![alt text](image_url) 或纯 URL
    markdown_image_regex = r'!\[.*?\]\((.*?)\)'
    url_regex = r'(https?://[^\s]+)'
    image_urls = re.findall(markdown_image_regex, text)
    if not image_urls:
        # 尝试匹配纯 URL
        image_urls = re.findall(url_regex, text)
    for url in image_urls:
        # 下载并保存图像
        await save_image(url)

async def save_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                content = await resp.read()
                # 从 URL 中获取文件名
                filename = url.split("/")[-1]
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    filename += '.png'  # 默认扩展名
                with open(filename, 'wb') as f:
                    f.write(content)
                print(f"已保存图片：{filename}")
            else:
                print("无法下载图片")

api_key = get_config('poe_api_key')
bot_name = "DALL-E-3"

asyncio.run(chat(api_key, bot_name))
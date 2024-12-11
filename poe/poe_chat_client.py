import requests

from common_utils import get_config


def chat_with_bot(user_input, api_key, bot_name):
    url = "http://8.137.126.154:8000/chat/"
    payload = {
        "user_input": user_input,
        "api_key": api_key,
        "bot_name": bot_name
    }
    response = requests.post(url, json=payload)
    return response.json()

if __name__ == "__main__":
    api_key = get_config('poe_api_key')
    bot_name = "GPT-3.5-Turbo"

    while True:
        user_input = input("你：")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break
        response = chat_with_bot(user_input, api_key, bot_name)
        print(f"{bot_name}：{response['response']}")
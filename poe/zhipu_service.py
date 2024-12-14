from flask import Flask, request, jsonify
import time
from zhipuai import ZhipuAI

app = Flask(__name__)

# 初始化ZhipuAI客户端
client = ZhipuAI(api_key="764d9c6af5fc87d7c35f9daedd9888cc.F3FAVkoMCA8BHjrj")

# 全局变量存储对话历史
conversation_history = []


def generate_response(user_input):
    """调用ZhipuAI模型生成响应"""
    # 将用户输入添加进对话历史
    conversation_history.append({"role": "user", "content": user_input})

    # 调用模型
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=conversation_history
    )

    assistant_reply = response.choices[0].message.content
    # 将AI回复也存储进历史
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Invalid input"}), 400
    try:
        start_time = time.time()
        response = generate_response(user_input)
        print("User:", user_input)
        print("Assistant:", response)
        print(f"Time elapsed: {time.time() - start_time}")
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

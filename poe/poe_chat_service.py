import fastapi
import fastapi_poe as fp
from pydantic import BaseModel
import os
# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

app = fastapi.FastAPI()

class ChatRequest(BaseModel):
    user_input: str
    api_key: str
    bot_name: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    messages = [fp.ProtocolMessage(role="user", content=request.user_input)]
    response_text = ""

    async for partial in fp.get_bot_response(messages=messages, bot_name=request.bot_name, api_key=request.api_key):
        if partial.text:
            response_text += partial.text

    messages.append(fp.ProtocolMessage(role="bot", content=response_text))
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
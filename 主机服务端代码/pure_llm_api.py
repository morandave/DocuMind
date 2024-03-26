'''接收prompt，将promt送入llm，返回回答'''
import os
from fastapi import FastAPI, Request
import json
import requests
import uvicorn

MODEL_PATH = os.environ.get('MODEL_PATH', '../chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

def chatGLM3(prompt):
    past_key_values, history = None, []
    query = "\n用户："+ prompt
    current_length = 0
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):        
        # print(response[current_length:], end="", flush=True)
        current_length = len(response)
    return response



app = FastAPI()
@app.post("/")
async def lc_llm(request: Request):
    #接收请求
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    print(prompt)
    #调用大模型
    final_response = chatGLM3(prompt)
    return final_response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
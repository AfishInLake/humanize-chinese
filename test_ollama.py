#! /usr/bin/env python
# @Time    : 2026/4/30 12:43
# @Author  : afish
# @File    : test_ollama.py
# test_ollama_api.py
import requests

OLLAMA_HOST = "http://192.168.0.188:11434"
MODEL_NAME = "qwen3.5:cloud"   # 改成你实际pull的模型名，例如 qwen2.5:3b 等

def test_connection():
    """测试 Ollama 服务是否在线"""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"[✅] 连接成功，服务器在线。已加载模型列表：{models}")
            return True
        else:
            print(f"[❌] 服务返回异常状态码：{resp.status_code}")
            return False
    except Exception as e:
        print(f"[❌] 无法连接到 Ollama 服务：{e}")
        return False

def test_generation():
    """测试一次简单的文本生成"""
    prompt = "你好，请用一句话介绍你自己"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "max_tokens": 50,
            "temperature": 0
        }
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("response", "")
            print(f"[✅] 模型生成成功。回复内容：\n{reply.strip()}")
            return True
        else:
            print(f"[❌] 生成请求失败，状态码：{resp.status_code}")
            return False
    except Exception as e:
        print(f"[❌] 生成请求出错：{e}")
        return False

if __name__ == "__main__":
    print("正在测试 Ollama API ...")
    if test_connection():
        test_generation()
#!/usr/bin/env python3
"""
测试本地 ONNX 模型（AnxForever/chinese-ai-detector-bert）
用法：
    python test_anx_onnx.py
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = Path("./bert_model_anx")  # 模型所在目录
ONNX_FILE = MODEL_DIR / "model.onnx"  # ONNX 模型文件（.data 自动加载）
TEMPERATURE = 0.8165  # 官方温度校准参数
MAX_LENGTH = 256  # 训练时的最大序列长度


def main():
    # 1. 加载分词器（本地文件）
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    # 2. 加载 ONNX 推理会话
    print("加载 ONNX 模型...")
    session = ort.InferenceSession(str(ONNX_FILE))
    print(f"  输入: {[inp.name for inp in session.get_inputs()]}")
    print(f"  输出: {[out.name for out in session.get_outputs()]}")

    # 3. 测试文本（你可以替换成自己的文本）
    test_texts = [
        "人工智能正在深刻改变社会的方方面面。",
        "今天天气真好，我和朋友去公园散步，看到了好多美丽的花。",
        "但这种方法要求所有机器人的运动能力基本相同，不适合本系统中轮式、履带式等异构平台共存的场景。",
        "深度神经网络通过多层非线性变换提取数据的高阶特征，从而实现复杂的模式识别任务。",
        "中午食堂的宫保鸡丁太咸了，下次换个窗口试试。",
    ]

    print("\n" + "=" * 60)
    print("开始推理...")
    print("=" * 60)
    for text in test_texts:
        # 分词编码
        enc = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)  # shape: (1, 256)
        attention_mask = enc["attention_mask"].astype(np.int64)

        # ONNX 推理
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        logits = outputs[0][0]  # 取出 batch 0 的 logits（形状 (2,)）

        # 温度校准 + softmax
        logits_scaled = logits / TEMPERATURE
        probs = np.exp(logits_scaled) / np.sum(np.exp(logits_scaled))

        # 判断结果
        pred_idx = int(np.argmax(probs))
        label = "AI" if pred_idx == 1 else "Human"
        human_prob = probs[0] * 100
        ai_prob = probs[1] * 100

        # 显示结果
        print(f"\n文本: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  预测: {label}  |  Human: {human_prob:.2f}%  |  AI: {ai_prob:.2f}%")

    # 4. 交互模式（可选）
    print("\n" + "=" * 60)
    print("进入交互模式（输入 quit 退出）")
    print("=" * 60)
    while True:
        try:
            user_input = input("\n请输入一段中文文本: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                break

            enc = tokenizer(
                user_input,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            outputs = session.run(None, {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })
            logits = outputs[0][0]
            logits_scaled = logits / TEMPERATURE
            probs = np.exp(logits_scaled) / np.sum(np.exp(logits_scaled))

            label = "AI" if np.argmax(probs) == 1 else "Human"
            print(f"  => {label}  (Human: {probs[0] * 100:.2f}% | AI: {probs[1] * 100:.2f}%)")

        except KeyboardInterrupt:
            print("\n退出。")
            break


if __name__ == "__main__":
    main()

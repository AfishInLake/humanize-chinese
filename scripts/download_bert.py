#!/usr/bin/env python3
"""
下载 bert-base-chinese 并导出两个 ONNX 模型。
不需要训练，直接用预训练权重。

产物：
  bert_base_chinese_mlm.onnx  — 句式评分器（~420MB）
  bert_base_chinese.onnx       — 语义保镖（~420MB）
  bert_base_chinese/           — tokenizer
"""

import os
import sys

def main():
    # ─── 检查依赖 ───
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
    except ImportError:
        print("请先安装: pip install torch")
        sys.exit(1)
    try:
        import transformers
        print(f"Transformers {transformers.__version__}")
    except ImportError:
        print("请先安装: pip install transformers")
        sys.exit(1)

    from transformers import BertTokenizer, BertModel, BertForMaskedLM

    MODEL = "bert-base-chinese"
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ─── 下载模型和 tokenizer ───
    print(f"\n[1/4] 下载 {MODEL} ...")
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    tokenizer.save_pretrained(os.path.join(OUT_DIR, "bert_base_chinese"))
    print(f"  Tokenizer 已保存")

    dummy = tokenizer("测试", return_tensors="pt", max_length=512,
                      truncation=True, padding="max_length")

    # ─── 导出 MLM 模型（句式评分器）───
    print(f"\n[2/4] 导出 MLM 模型（句式评分器）...")
    mlm = BertForMaskedLM.from_pretrained(MODEL)
    mlm.eval()

    mlm_path = os.path.join(OUT_DIR, "bert_base_chinese_mlm.onnx")
    torch.onnx.export(
        mlm,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        mlm_path,
        dynamo=False,  # 关键！用旧版导出器，确保权重完整嵌入
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "output": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size = os.path.getsize(mlm_path)
    ok = "✅" if size > 100_000_000 else "❌ 太小！"
    print(f"  {ok} 已保存: {mlm_path} ({size/1e6:.0f}MB)")

    # ─── 导出 BertModel（语义保镖）───
    print(f"\n[3/4] 导出 BertModel（语义保镖）...")
    bert = BertModel.from_pretrained(MODEL)
    bert.eval()

    bert_path = os.path.join(OUT_DIR, "bert_base_chinese.onnx")
    torch.onnx.export(
        bert,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        bert_path,
        dynamo=False,  # 关键！用旧版导出器，确保权重完整嵌入
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "output": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size = os.path.getsize(bert_path)
    ok = "✅" if size > 100_000_000 else "❌ 太小！"
    print(f"  {ok} 已保存: {bert_path} ({size/1e6:.0f}MB)")

    # ─── 完成 ───
    print(f"\n[4/4] 完成！产物：")
    print(f"  {OUT_DIR}/bert_base_chinese_mlm.onnx  (句式评分)")
    print(f"  {OUT_DIR}/bert_base_chinese.onnx       (语义保镖)")
    print(f"  {OUT_DIR}/bert_base_chinese/           (tokenizer)")
    print(f"\n  三个功能全部就绪，不需要训练。")

if __name__ == "__main__":
    main()

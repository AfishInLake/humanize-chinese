#!/usr/bin/env python3
"""
一键导出所有 BERT ONNX 模型。

在你的 Mac 上运行（需要已安装 torch + transformers + onnx）：

    source ~/bert-env/bin/activate
    python3 export_all_bert.py

导出产物：
  1. bert_base_chinese_mlm.onnx  — 句式评分器用（MLM logits）
  2. bert_base_chinese.onnx       — 语义保镖用（hidden states）
  3. bert_base_chinese/           — tokenizer（共用）
"""

import os
import sys
import shutil

# 确保能找到同目录的 export_onnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_onnx import load_model, export_to_onnx

MODEL_NAME = "bert-base-chinese"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def export_mlm():
    """导出 BertForMaskedLM — 句式评分器用。"""
    print("\n" + "=" * 60)
    print("  [1/2] 导出 MLM 模型（句式评分器）")
    print("=" * 60)

    model, tokenizer = load_model(MODEL_NAME)

    output_path = os.path.join(SCRIPT_DIR, "bert_base_chinese_mlm.onnx")
    export_to_onnx(model, output_path, max_length=512)

    # 保存 tokenizer
    tokenizer_dir = os.path.join(SCRIPT_DIR, "bert_base_chinese")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"[Tokenizer] 已保存到: {tokenizer_dir}")

    return tokenizer


def export_bert_model():
    """导出 BertModel — 语义保镖用（hidden states）。"""
    print("\n" + "=" * 60)
    print("  [2/2] 导出 BertModel（语义保镖）")
    print("=" * 60)

    from transformers import BertModel, BertTokenizer
    import torch

    print(f"[加载] 从 {MODEL_NAME} 加载 BertModel...")
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[加载] 参数量: {total_params:,}")

    output_path = os.path.join(SCRIPT_DIR, "bert_base_chinese.onnx")

    # Dummy input
    dummy = tokenizer("测试文本", return_tensors="pt", max_length=512,
                      truncation=True, padding='max_length')

    print(f"[导出] 输出路径: {output_path}")
    torch.onnx.export(
        model,
        (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids']),
        output_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_length'},
            'attention_mask': {0: 'batch_size', 1: 'seq_length'},
            'token_type_ids': {0: 'batch_size', 1: 'seq_length'},
            'output': {0: 'batch_size', 1: 'seq_length'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size = os.path.getsize(output_path)
    print(f"[导出] 完成! 大小: {size / 1e6:.1f} MB")

    return tokenizer


def main():
    print("=" * 60)
    print("  BERT ONNX 模型一键导出")
    print("=" * 60)
    print(f"  源模型: {MODEL_NAME}")
    print(f"  输出目录: {SCRIPT_DIR}")
    print()

    # 检查依赖
    try:
        import torch
        print(f"[环境] PyTorch {torch.__version__}")
    except ImportError:
        print("[错误] 未安装 PyTorch，请先运行: pip install torch")
        sys.exit(1)

    try:
        import transformers
        print(f"[环境] Transformers {transformers.__version__}")
    except ImportError:
        print("[错误] 未安装 Transformers，请先运行: pip install transformers")
        sys.exit(1)

    try:
        import onnx
        print(f"[环境] ONNX {onnx.__version__}")
    except ImportError:
        print("[错误] 未安装 ONNX，请先运行: pip install onnx")
        sys.exit(1)

    # 导出
    tokenizer = export_mlm()
    export_bert_model()

    # 汇总
    print("\n" + "=" * 60)
    print("  导出完成！")
    print("=" * 60)
    print()
    print("  产物文件：")
    print(f"    {SCRIPT_DIR}/bert_base_chinese_mlm.onnx  (句式评分器)")
    print(f"    {SCRIPT_DIR}/bert_base_chinese.onnx       (语义保镖)")
    print(f"    {SCRIPT_DIR}/bert_base_chinese/           (tokenizer)")
    print()
    print("  下一步：把这些文件复制到你的项目的 scripts/ 目录即可。")
    print("  句式评分器和语义保镖会自动检测并加载。")


if __name__ == '__main__':
    main()

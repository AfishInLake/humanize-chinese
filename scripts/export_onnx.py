#!/usr/bin/env python3
"""
BERT 模型 ONNX 导出脚本。

将 HuggingFace 格式的 BERT 模型导出为 ONNX 格式，
供本机 semantic_guard.py 使用。

用法:
    # 基本导出
    python export_onnx.py --model_dir ./bert_model --output ./scripts/bert_base_chinese.onnx

    # FP16 量化导出
    python export_onnx.py --model_dir ./bert_model --output ./scripts/bert_base_chinese.onnx --fp16

    # 指定最大序列长度
    python export_onnx.py --model_dir ./bert_model --output ./scripts/bert_base_chinese.onnx --max_length 256

依赖安装:
    pip install torch transformers onnx onnxruntime
"""

import os
import sys
import argparse
import time

import numpy as np
import torch


def load_model(model_dir):
    """从 HuggingFace 目录加载 BERT 模型。

    Args:
        model_dir: HuggingFace 模型目录路径

    Returns:
        (model, tokenizer) 元组
    """
    from transformers import BertForSequenceClassification, BertTokenizer

    print(f"[加载] 从 {model_dir} 加载模型...")
    if not os.path.isdir(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)

    # 检查必要文件
    required_files = ["config.json", "model.safetensors"]
    optional_files = ["pytorch_model.bin"]
    has_required = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
    has_optional = any(os.path.exists(os.path.join(model_dir, f)) for f in optional_files)

    if not has_required and not has_optional:
        print(f"错误: 在 {model_dir} 中未找到模型权重文件")
        print(f"  需要: {required_files} 或 {optional_files}")
        sys.exit(1)

    # 加载模型和 tokenizer
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    model.eval()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[加载] 模型加载完成")
    print(f"[加载] 参数量: {total_params:,}")
    print(f"[加载] 标签映射: {model.config.id2label}")

    return model, tokenizer


def export_to_onnx(model, output_path, max_length=512, opset_version=14):
    """将 PyTorch 模型导出为 ONNX 格式。

    Args:
        model:        PyTorch 模型
        output_path:  ONNX 输出路径
        max_length:   最大序列长度（用于 dummy input）
        opset_version: ONNX opset 版本
    """
    import onnx

    print(f"\n[导出] 开始 ONNX 导出...")
    print(f"[导出] 输出路径: {output_path}")
    print(f"[导出] 最大序列长度: {max_length}")
    print(f"[导出] opset 版本: {opset_version}")

    # 创建 dummy 输入
    dummy_input_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

    # 动态轴设置：支持可变的 batch_size 和 seq_length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "output": {0: "batch_size"},
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 执行导出
    start_time = time.time()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    export_time = time.time() - start_time

    print(f"[导出] 导出完成，耗时 {export_time:.1f}s")

    # 验证 ONNX 模型
    print(f"[验证] 检查 ONNX 模型结构...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[验证] 模型结构验证通过")

    # 打印模型图信息
    graph = onnx_model.graph
    print(f"[验证] 输入节点:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else "dynamic" for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    print(f"[验证] 输出节点:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else "dynamic" for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    return output_path


def quantize_onnx(input_path, output_path=None):
    """对 ONNX 模型执行动态量化。

    使用 uint8 量化权重，减小模型体积。

    Args:
        input_path:  原始 ONNX 模型路径
        output_path: 量化后输出路径（默认在原文件名后加 _quantized）

    Returns:
        量化后的模型路径
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("错误: 未安装 onnxruntime。请运行: pip install onnxruntime")
        sys.exit(1)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_quantized{ext}"

    print(f"\n[量化] 开始动态量化...")
    print(f"[量化] 输入: {input_path}")
    print(f"[量化] 输出: {output_path}")

    start_time = time.time()
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )
    quant_time = time.time() - start_time

    print(f"[量化] 量化完成，耗时 {quant_time:.1f}s")

    return output_path


def verify_onnx_runtime(onnx_path, tokenizer, max_length=512):
    """使用 ONNX Runtime 验证导出的模型。

    Args:
        onnx_path:   ONNX 模型路径
        tokenizer:   BertTokenizer 实例
        max_length:  最大序列长度
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[验证] 跳过 ONNX Runtime 验证（未安装 onnxruntime）")
        return

    print(f"\n[推理验证] 使用 ONNX Runtime 进行推理测试...")

    # 创建推理会话
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options)

    # 获取输入输出信息
    print(f"[推理验证] 模型输入:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
    print(f"[推理验证] 模型输出:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    # 测试用例
    test_texts = [
        "这是一段人类撰写的测试文本，用于验证模型的推理功能。",
        "人工智能生成的文本通常具有特定的语言模式和结构特征。",
    ]

    print(f"\n[推理验证] 测试 {len(test_texts)} 条文本:")
    for i, text in enumerate(test_texts):
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        # 推理
        start_time = time.time()
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        inference_time = (time.time() - start_time) * 1000  # 毫秒

        # 解析结果
        logits = outputs[0]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        pred_class = np.argmax(logits, axis=-1)[0]
        confidence = probs[0][pred_class]

        label_map = {0: "Human", 1: "AI"}
        pred_label = label_map.get(pred_class, f"Unknown({pred_class})")

        print(f"  [{i+1}] 预测: {pred_label} (置信度: {confidence:.4f})")
        print(f"      耗时: {inference_time:.1f}ms")
        print(f"      文本: {text[:50]}...")

    print(f"\n[推理验证] ONNX Runtime 验证完成")


def print_size_comparison(original_path, quantized_path=None):
    """打印模型大小对比。

    Args:
        original_path:   原始模型路径
        quantized_path:  量化模型路径（可选）
    """
    print(f"\n{'='*50}")
    print("模型大小对比")
    print(f"{'='*50}")

    original_size = os.path.getsize(original_path)
    print(f"  原始模型:     {original_size / 1e6:.2f} MB ({original_path})")

    if quantized_path and os.path.exists(quantized_path):
        quantized_size = os.path.getsize(quantized_path)
        ratio = original_size / quantized_size
        reduction = (1 - quantized_size / original_size) * 100
        print(f"  量化模型:     {quantized_size / 1e6:.2f} MB ({quantized_path})")
        print(f"  压缩比:       {ratio:.2f}x")
        print(f"  体积减少:     {reduction:.1f}%")

    print(f"{'='*50}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="BERT 模型 ONNX 导出脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本导出
  python export_onnx.py --model_dir ./bert_model --output ./scripts/bert_base_chinese.onnx

  # FP16 量化导出
  python export_onnx.py --model_dir ./bert_model --output ./scripts/bert_base_chinese.onnx --fp16

  # 自定义序列长度
  python export_onnx.py --model_dir ./bert_model --output ./scripts/bert_base_chinese.onnx --max_length 256
        """,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="HuggingFace 模型目录路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="ONNX 输出路径 (默认: <model_dir>/bert_base_chinese.onnx)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度 (默认: 512)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset 版本 (默认: 14)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="使用动态量化 (uint8) 减小模型体积 (默认: False)",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        default=False,
        help="跳过 ONNX Runtime 推理验证 (默认: False)",
    )

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        args.output = os.path.join(args.model_dir, "bert_base_chinese.onnx")

    print(f"{'='*60}")
    print("BERT 模型 ONNX 导出工具")
    print(f"{'='*60}")
    print(f"  模型目录:     {args.model_dir}")
    print(f"  输出路径:     {args.output}")
    print(f"  最大长度:     {args.max_length}")
    print(f"  opset 版本:   {args.opset}")
    print(f"  动态量化:     {args.fp16}")
    print(f"{'='*60}\n")

    # ─── 检查依赖 ───
    try:
        import onnx
        print(f"[环境] ONNX 版本: {onnx.__version__}")
    except ImportError:
        print("错误: 未安装 ONNX。请运行: pip install onnx")
        sys.exit(1)

    try:
        import transformers
        print(f"[环境] Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("错误: 未安装 Transformers。请运行: pip install transformers")
        sys.exit(1)

    print(f"[环境] PyTorch 版本: {torch.__version__}")

    # ─── 步骤 1: 加载模型 ───
    model, tokenizer = load_model(args.model_dir)

    # ─── 步骤 2: 导出 ONNX ───
    onnx_path = export_to_onnx(
        model=model,
        output_path=args.output,
        max_length=args.max_length,
        opset_version=args.opset,
    )

    # ─── 步骤 3: 量化（可选）───
    quantized_path = None
    if args.fp16:
        quantized_path = quantize_onnx(onnx_path)
        # 用量化版本替换原始版本
        original_size = os.path.getsize(onnx_path)
        quantized_size = os.path.getsize(quantized_path)
        print(f"[量化] 原始大小:   {original_size / 1e6:.2f} MB")
        print(f"[量化] 量化后大小: {quantized_size / 1e6:.2f} MB")
        print(f"[量化] 压缩比:     {original_size / quantized_size:.2f}x")

        # 替换原始文件
        os.replace(quantized_path, onnx_path)
        print(f"[量化] 已用量化模型替换原始模型")
        quantized_path = None  # 已替换，不再需要对比

    # ─── 步骤 4: 验证推理 ───
    if not args.skip_verify:
        verify_onnx_runtime(onnx_path, tokenizer, args.max_length)

    # ─── 步骤 5: 打印大小信息 ───
    final_size = os.path.getsize(onnx_path)
    print(f"\n{'='*60}")
    print("导出完成!")
    print(f"{'='*60}")
    print(f"  ONNX 模型路径: {onnx_path}")
    print(f"  模型大小:      {final_size / 1e6:.2f} MB")
    print(f"\n  下一步: 将 {onnx_path} 复制到")
    print(f"         humanize-chinese/scripts/ 目录即可使用。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BERT 中文 AI 文本检测器训练脚本。

在另一台有 GPU 的机器上运行，训练完成后导出 ONNX 模型，
复制到本机 scripts/ 目录即可使用。

用法:
    # 完整训练
    python train_bert_detector.py --data_dir ./training_data --output_dir ./bert_model

    # 快速测试（100 条数据）
    python train_bert_detector.py --data_dir ./training_data --output_dir ./bert_model --quick

    # 导出 FP16 量化 ONNX
    python train_bert_detector.py --data_dir ./training_data --output_dir ./bert_model --fp16

数据准备:
    在 data_dir 下创建两个子目录:
    - ai_texts/    : AI 生成的文本文件（每行一段）
    - human_texts/ : 人类撰写的文本文件（每行一段）

    或放置一个 JSON 文件:
    - data.jsonl   : 每行 {"text": "...", "label": 0 或 1}

    或放置一个 CSV 文件:
    - data.csv     : 包含 text, label 两列

依赖安装（训练机）:
    pip install torch transformers onnx onnxruntime datasets scikit-learn
"""

import os
import sys
import json
import argparse
import random
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── 忽略部分警告 ───
warnings.filterwarnings("ignore", category=UserWarning)

# ─── PyTorch / Transformers ───
import torch
from torch.utils.data import Dataset

# ─── sklearn 评估指标 ───
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# 数据集类
# ============================================================

class TextClassificationDataset(Dataset):
    """中文文本二分类数据集。

    将文本通过 BertTokenizer 编码为模型输入格式。
    """

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts:   文本列表
            labels:  标签列表（0=人类, 1=AI）
            tokenizer: BertTokenizer 实例
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # 使用 tokenizer 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 数据加载与清洗
# ============================================================

def clean_text(text):
    """清洗单条文本。

    - 去除首尾空白
    - 合并连续空白字符为单个空格
    - 去除零宽字符等不可见字符
    """
    if not text:
        return ""
    # 去除零宽字符
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    # 合并连续空白
    import re
    text = re.sub(r"\s+", " ", text.strip())
    return text


def load_data_from_jsonl(filepath):
    """从 JSONL 文件加载数据。

    每行格式: {"text": "...", "label": 0 或 1}

    Returns:
        (texts, labels) 元组
    """
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text = row.get("text", "")
                label = row.get("label")
                if text and label is not None:
                    texts.append(text)
                    labels.append(int(label))
            except json.JSONDecodeError:
                continue
    return texts, labels


def load_data_from_csv(filepath):
    """从 CSV 文件加载数据。

    期望包含 text 和 label 两列。

    Returns:
        (texts, labels) 元组
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath, encoding="utf-8")
        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        return texts, labels
    except ImportError:
        # pandas 不可用时，手动解析 CSV
        texts, labels = [], []
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline()  # 跳过表头
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    texts.append(parts[1])
                    labels.append(int(parts[0]))
                elif len(parts) == 1:
                    # 尝试以 tab 分隔
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        texts.append(parts[1])
                        labels.append(int(parts[0]))
        return texts, labels


def load_data_from_directories(data_dir):
    """从目录结构加载数据。

    期望目录结构:
        data_dir/
            ai_texts/     : AI 生成的文本文件
            human_texts/  : 人类撰写的文本文件

    每个文件中，每行视为一条独立文本。

    Returns:
        (texts, labels) 元组
    """
    texts, labels = [], []

    # 加载 AI 文本
    ai_dir = os.path.join(data_dir, "ai_texts")
    if os.path.isdir(ai_dir):
        for fname in os.listdir(ai_dir):
            fpath = os.path.join(ai_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        labels.append(1)  # 1 = AI

    # 加载人类文本
    human_dir = os.path.join(data_dir, "human_texts")
    if os.path.isdir(human_dir):
        for fname in os.listdir(human_dir):
            fpath = os.path.join(human_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        labels.append(0)  # 0 = Human

    return texts, labels


def load_data_from_hc3_jsonl(filepath):
    """从 HC3-Chinese 原始格式 JSONL 加载数据。

    HC3 格式每行:
    {
        "question": "...",
        "chatgpt_answers": ["...", "..."],
        "human_answers": ["...", "..."]
    }

    Returns:
        (texts, labels) 元组
    """
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # AI 生成的回答
                for ans in row.get("chatgpt_answers", []) or []:
                    if ans and len(ans.strip()) > 10:
                        texts.append(ans.strip())
                        labels.append(1)
                # 人类撰写的回答
                for ans in row.get("human_answers", []) or []:
                    if ans and len(ans.strip()) > 10:
                        texts.append(ans.strip())
                        labels.append(0)
            except json.JSONDecodeError:
                continue
    return texts, labels


def load_all_data(data_dir, min_chars=10):
    """从 data_dir 加载所有可用数据。

    按以下优先级尝试:
    1. data.jsonl（标准格式）
    2. data.csv
    3. ai_texts/ + human_texts/ 目录结构
    4. hc3_all.jsonl（HC3 原始格式）

    Args:
        data_dir:   数据目录路径
        min_chars:  文本最小字符数（低于此值的文本将被过滤）

    Returns:
        (texts, labels) 清洗后的数据
    """
    data_path = Path(data_dir)

    # 尝试标准 JSONL 格式
    jsonl_path = data_path / "data.jsonl"
    if jsonl_path.exists():
        print(f"[数据] 从 {jsonl_path} 加载标准 JSONL 格式...")
        texts, labels = load_data_from_jsonl(str(jsonl_path))

    # 尝试 CSV 格式
    elif (data_path / "data.csv").exists():
        print(f"[数据] 从 {data_path / 'data.csv'} 加载 CSV 格式...")
        texts, labels = load_data_from_csv(str(data_path / "data.csv"))

    # 尝试 HC3 原始格式
    elif (data_path / "hc3_all.jsonl").exists():
        print(f"[数据] 从 {data_path / 'hc3_all.jsonl'} 加载 HC3 原始格式...")
        texts, labels = load_data_from_hc3_jsonl(str(data_path / "hc3_all.jsonl"))

    # 尝试目录结构
    elif (data_path / "ai_texts").is_dir() or (data_path / "human_texts").is_dir():
        print(f"[数据] 从目录结构加载 (ai_texts/ + human_texts/)...")
        texts, labels = load_data_from_directories(str(data_path))

    else:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到可用数据。\n"
            f"请确保存在以下之一:\n"
            f"  - data.jsonl  (每行 {{\"text\": \"...\", \"label\": 0/1}})\n"
            f"  - data.csv    (text, label 两列)\n"
            f"  - hc3_all.jsonl (HC3 原始格式)\n"
            f"  - ai_texts/ + human_texts/ 目录"
        )

    # 数据清洗
    print(f"[数据] 原始数据量: {len(texts)} 条")
    cleaned_texts, cleaned_labels = [], []
    for text, label in zip(texts, labels):
        text = clean_text(text)
        if len(text) >= min_chars:
            cleaned_texts.append(text)
            cleaned_labels.append(label)

    n_removed = len(texts) - len(cleaned_texts)
    if n_removed > 0:
        print(f"[数据] 清洗后移除 {n_removed} 条（空文本或过短）")

    # 统计类别分布
    n_ai = sum(1 for l in cleaned_labels if l == 1)
    n_human = sum(1 for l in cleaned_labels if l == 0)
    print(f"[数据] 清洗后数据量: {len(cleaned_texts)} 条 (AI: {n_ai}, Human: {n_human})")

    return cleaned_texts, cleaned_labels


# ============================================================
# 模型训练
# ============================================================

def compute_metrics(eval_pred):
    """计算评估指标，供 HuggingFace Trainer 使用。

    Args:
        eval_pred: EvalPrediction 对象，包含 predictions 和 label_ids

    Returns:
        指标字典
    """
    from transformers import EvalPrediction

    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # 处理多输出情况（取 logits）
    if isinstance(logits, tuple):
        logits = logits[0]

    # 计算预测概率和类别
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    if logits.shape[-1] == 2:
        # 二分类，取正类概率
        preds = np.argmax(logits, axis=-1)
        pos_probs = probs[:, 1] if probs.ndim > 1 else probs
    else:
        preds = (probs > 0.5).astype(int)
        pos_probs = probs

    # 计算各项指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # AUC（需要正类概率）
    try:
        auc = roc_auc_score(labels, pos_probs)
    except ValueError:
        # 当只有一个类别时无法计算 AUC
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


class EarlyStoppingCallback:
    """早停回调：当 eval loss 连续 patience 轮未改善时停止训练。"""

    def __init__(self, patience=2, threshold=1e-4):
        """
        Args:
            patience:   允许的最大未改善轮数
            threshold:  改善的最小阈值
        """
        self.patience = patience
        self.threshold = threshold
        self.best_loss = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """在每次评估后调用。"""
        current_loss = metrics.get("eval_loss") if metrics else None
        if current_loss is None:
            return

        if self.best_loss is None:
            self.best_loss = current_loss
            return

        # 检查是否有改善
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"[早停] eval_loss 未改善 ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                print(f"[早停] 连续 {self.patience} 轮未改善，停止训练")
                control.should_training_stop = True


def train_model(
    data_dir,
    output_dir,
    epochs=3,
    batch_size=16,
    max_length=512,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=42,
    quick=False,
    export_onnx=True,
    fp16_onnx=False,
):
    """训练 BERT 中文 AI 文本检测器。

    Args:
        data_dir:      训练数据目录
        output_dir:    模型输出目录
        epochs:        训练轮数
        batch_size:    批大小
        max_length:    最大序列长度
        learning_rate: 学习率
        warmup_ratio:  预热比例
        weight_decay:  权重衰减
        seed:          随机种子
        quick:         快速测试模式
        export_onnx:   是否导出 ONNX
        fp16_onnx:     是否使用 FP16 量化
    """
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    # ─── 设置随机种子 ───
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ─── 设备信息 ───
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] 使用 {device}")
    if torch.cuda.is_available():
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[设备] GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ─── 加载数据 ───
    print(f"\n{'='*60}")
    print("[步骤 1/5] 加载训练数据")
    print(f"{'='*60}")
    texts, labels = load_all_data(data_dir)

    # 快速测试模式：每类只取 50 条
    if quick:
        ai_indices = [i for i, l in enumerate(labels) if l == 1]
        human_indices = [i for i, l in enumerate(labels) if l == 0]
        rng = random.Random(seed)
        rng.shuffle(ai_indices)
        rng.shuffle(human_indices)
        n_per_class = min(50, len(ai_indices), len(human_indices))
        selected = ai_indices[:n_per_class] + human_indices[:n_per_class]
        texts = [texts[i] for i in selected]
        labels = [labels[i] for i in selected]
        print(f"[快速模式] 采样 {len(texts)} 条数据用于快速测试")

    # ─── 数据划分 ───
    print(f"\n{'='*60}")
    print("[步骤 2/5] 划分训练集 / 测试集")
    print(f"{'='*60}")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    print(f"训练集: {len(train_texts)} 条")
    print(f"测试集: {len(test_texts)} 条")

    # ─── 加载 Tokenizer 和模型 ───
    print(f"\n{'='*60}")
    print("[步骤 3/5] 初始化模型")
    print(f"{'='*60}")

    model_name = "bert-base-chinese"
    print(f"[模型] 加载预训练模型: {model_name}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1},
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数量: {total_params:,}")
    print(f"[模型] 可训练参数量: {trainable_params:,}")

    # ─── 创建数据集 ───
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, max_length
    )

    # ─── 训练参数 ───
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_dir=str(output_path / "logs"),
        logging_steps=50,
        # 评估策略：每个 epoch 结束后评估
        eval_strategy="epoch",
        # 保存策略：每个 epoch 结束后保存
        save_strategy="epoch",
        # 只保留最好的模型
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        # 性能优化
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        # 避免警告
        report_to="none",
        seed=seed,
        # 禁用 tqdm 以减少输出（训练机上通常不需要）
        # disable_tqdm=False,
    )

    # ─── 初始化 Trainer ───
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=2)],
    )

    # ─── 开始训练 ───
    print(f"\n{'='*60}")
    print("[步骤 4/5] 开始训练")
    print(f"{'='*60}")
    print(f"  学习率:     {learning_rate}")
    print(f"  批大小:     {batch_size}")
    print(f"  训练轮数:   {epochs}")
    print(f"  最大长度:   {max_length}")
    print(f"  预热比例:   {warmup_ratio}")
    print(f"  权重衰减:   {weight_decay}")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    # ─── 保存最终模型 ───
    print(f"\n[保存] 保存最终模型到 {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # 保存训练元信息
    meta = {
        "model_name": model_name,
        "trained_at": datetime.now().isoformat(),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "train_loss": float(train_result.training_loss),
        "seed": seed,
    }
    with open(output_path / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ─── 评估 ───
    print(f"\n{'='*60}")
    print("[步骤 5/5] 模型评估")
    print(f"{'='*60}")

    eval_results = trainer.evaluate()
    print(f"\n评估结果:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # 详细评估
    print(f"\n{'='*60}")
    print("详细分类报告")
    print(f"{'='*60}")

    # 获取测试集预测结果
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    true_labels = predictions.label_ids

    # 分类报告
    report = classification_report(
        true_labels,
        preds,
        target_names=["Human", "AI"],
        digits=4,
    )
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(true_labels, preds)
    print("混淆矩阵:")
    print(f"              预测Human  预测AI")
    print(f"  实际Human    {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  实际AI       {cm[1][0]:>6}    {cm[1][1]:>6}")

    # AUC 分数
    if logits.shape[-1] == 2:
        probs = 1.0 / (1.0 + np.exp(-logits))
        pos_probs = probs[:, 1]
    else:
        pos_probs = 1.0 / (1.0 + np.exp(-logits))

    try:
        auc = roc_auc_score(true_labels, pos_probs)
        print(f"\nROC AUC: {auc:.4f}")
    except ValueError:
        print("\nROC AUC: 无法计算（测试集中只有一个类别）")

    # 保存评估结果
    eval_meta = {
        "eval_results": {k: float(v) if isinstance(v, (int, float)) else v
                         for k, v in eval_results.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(output_path / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_meta, f, indent=2, ensure_ascii=False)

    # ─── ONNX 导出 ───
    if export_onnx:
        print(f"\n{'='*60}")
        print("导出 ONNX 模型")
        print(f"{'='*60}")
        export_to_onnx(
            model_dir=str(output_path),
            output_path=str(output_path / "bert_base_chinese.onnx"),
            fp16=fp16_onnx,
            max_length=max_length,
        )

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"模型保存位置: {output_path}")
    if export_onnx:
        print(f"ONNX 模型:   {output_path / 'bert_base_chinese.onnx'}")
    print(f"\n下一步: 将 {output_path / 'bert_base_chinese.onnx'} 复制到")
    print(f"       humanize-chinese/scripts/ 目录即可使用。")


# ============================================================
# ONNX 导出
# ============================================================

def export_to_onnx(model_dir, output_path, fp16=False, max_length=512):
    """将 HuggingFace BERT 模型导出为 ONNX 格式。

    Args:
        model_dir:    HuggingFace 模型目录
        output_path:  ONNX 输出路径
        fp16:         是否使用 FP16 量化
        max_length:   最大序列长度（用于 dummy input）
    """
    import onnx
    from transformers import BertForSequenceClassification

    print(f"[ONNX] 加载模型: {model_dir}")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # 创建 dummy 输入
    dummy_input_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

    # 动态轴设置：支持可变的 batch_size 和 seq_length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "output": {0: "batch_size"},
    }

    print(f"[ONNX] 导出中... (FP16: {fp16})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    # 验证导出的模型
    print(f"[ONNX] 验证模型...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[ONNX] 模型验证通过")

    # FP16 量化
    if fp16:
        print(f"[ONNX] 执行 FP16 量化...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        fp16_path = output_path.replace(".onnx", "_fp16.onnx")
        quantize_dynamic(
            model_input=output_path,
            model_output=fp16_path,
            weight_type=QuantType.QUInt8,
        )
        # 更新输出路径为量化后的模型
        original_size = os.path.getsize(output_path)
        quantized_size = os.path.getsize(fp16_path)
        print(f"[ONNX] FP16 量化完成")
        print(f"[ONNX] 原始大小:   {original_size / 1e6:.2f} MB")
        print(f"[ONNX] 量化后大小: {quantized_size / 1e6:.2f} MB")
        print(f"[ONNX] 压缩比:     {original_size / quantized_size:.2f}x")

        # 用量化版本替换原始版本
        os.replace(fp16_path, output_path)

    # 打印最终模型大小
    final_size = os.path.getsize(output_path)
    print(f"[ONNX] 最终模型大小: {final_size / 1e6:.2f} MB")
    print(f"[ONNX] 保存到: {output_path}")

    # 验证 ONNX Runtime 推理
    print(f"[ONNX] 验证 ONNX Runtime 推理...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path)
        # 使用简单输入测试
        test_input_ids = np.zeros((1, 32), dtype=np.int64)
        test_attention_mask = np.ones((1, 32), dtype=np.int64)
        outputs = sess.run(
            None,
            {
                "input_ids": test_input_ids,
                "attention_mask": test_attention_mask,
            },
        )
        print(f"[ONNX] 推理测试通过，输出形状: {outputs[0].shape}")
    except ImportError:
        print(f"[ONNX] 跳过 ONNX Runtime 推理测试（未安装 onnxruntime）")
    except Exception as e:
        print(f"[ONNX] 推理测试失败: {e}")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="BERT 中文 AI 文本检测器训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整训练
  python train_bert_detector.py --data_dir ./training_data --output_dir ./bert_model

  # 快速测试
  python train_bert_detector.py --data_dir ./training_data --output_dir ./bert_model --quick

  # FP16 量化导出
  python train_bert_detector.py --data_dir ./training_data --output_dir ./bert_model --fp16
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="训练数据目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bert_model",
        help="模型输出目录 (默认: ./bert_model)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数 (默认: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批大小 (默认: 16)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度 (默认: 512)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="学习率 (默认: 2e-5)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="学习率预热比例 (默认: 0.1)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减系数 (默认: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        default=True,
        help="训练完成后导出 ONNX 模型 (默认: True)",
    )
    parser.add_argument(
        "--no_export_onnx",
        action="store_false",
        dest="export_onnx",
        help="不导出 ONNX 模型",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="使用 FP16 量化导出 ONNX (默认: False)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="快速测试模式，使用少量数据 (默认: False)",
    )

    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    print(f"{'='*60}")
    print("BERT 中文 AI 文本检测器 - 训练脚本")
    print(f"{'='*60}")
    print(f"  数据目录:   {args.data_dir}")
    print(f"  输出目录:   {args.output_dir}")
    print(f"  训练轮数:   {args.epochs}")
    print(f"  批大小:     {args.batch_size}")
    print(f"  最大长度:   {args.max_length}")
    print(f"  学习率:     {args.learning_rate}")
    print(f"  快速模式:   {args.quick}")
    print(f"  FP16 量化:  {args.fp16}")
    print(f"{'='*60}\n")

    # 检查数据目录
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)

    # 检查 PyTorch
    try:
        import torch
        print(f"[环境] PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("错误: 未安装 PyTorch。请运行: pip install torch")
        sys.exit(1)

    # 检查 Transformers
    try:
        import transformers
        print(f"[环境] Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("错误: 未安装 Transformers。请运行: pip install transformers")
        sys.exit(1)

    # 检查 ONNX（仅在需要导出时）
    if args.export_onnx:
        try:
            import onnx
            print(f"[环境] ONNX 版本: {onnx.__version__}")
        except ImportError:
            print("警告: 未安装 ONNX。训练将继续，但不会导出 ONNX 模型。")
            args.export_onnx = False

    # 开始训练
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        quick=args.quick,
        export_onnx=args.export_onnx,
        fp16_onnx=args.fp16,
    )


if __name__ == "__main__":
    main()

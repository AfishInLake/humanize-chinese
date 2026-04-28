#!/usr/bin/env python3
"""
BERT AI 文本检测器 — 基于 ONNX 推理。

使用用户训练的二分类 BERT 模型（human / ai），
输出 0-100 的 AI 概率分数。

当前集成模型: AnxForever/chinese-ai-detector-bert
ONNX 模型位于 bert_model_anx/ 目录。
模型不可用时自动降级为跳过（返回 None）。

使用方式：
    from bert_detector import bert_detect_score
    score = bert_detect_score("一段文本")  # 返回 0-100 或 None
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ─── 配置加载 ───
from ..config import load_config as _load_cfg
_CFG = _load_cfg()
_BDCFG = _CFG.get('bert_detector', {})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 全局状态（延迟初始化）───
_tokenizer = None
_onnx_session = None
_available = None  # None=未检测, True/False=已检测
_label_map = None  # id -> label 映射


def _init():
    """延迟加载 tokenizer + ONNX 模型。

    Returns:
        bool: 是否加载成功
    """
    global _tokenizer, _onnx_session, _available, _label_map

    if _available is not None:
        return _available

    enabled = _BDCFG.get('enabled', True)
    if not enabled:
        _available = False
        return False

    # 模型路径
    model_dir_name = _BDCFG.get('model_dir', 'detector')
    model_dir = os.path.join(SCRIPT_DIR, '..', 'data', 'models', model_dir_name)
    tokenizer_dir = os.path.join(SCRIPT_DIR, '..', 'data')
    onnx_name = _BDCFG.get('onnx_model_path', 'model.onnx')
    onnx_path = os.path.join(model_dir, onnx_name)

    if not os.path.exists(onnx_path):
        logger.info("BERT 检测器 ONNX 模型不存在: %s，将跳过 BERT 检测", onnx_path)
        _available = False
        return False

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # 加载 tokenizer（从 data/ 目录）
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

        # 加载 ONNX 模型（分片格式 model.onnx + model.onnx.data 自动处理）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _onnx_session = ort.InferenceSession(onnx_path, sess_options)

        # 解析 label 映射
        _label_map = _parse_label_map(model_dir)

        _available = True
        logger.info("BERT 检测器加载成功: %s", onnx_path)

    except ImportError:
        _available = False
        logger.info("onnxruntime/transformers 未安装，BERT 检测不可用")
    except Exception as e:
        _available = False
        logger.warning("BERT 检测器加载失败: %s", e)

    return _available


def _parse_label_map(model_dir):
    """解析 label 映射。

    优先从 config.json 读取 id2label，
    如果没有则使用配置文件中的默认值。

    Returns:
        dict: {0: "Human", 1: "AI"} 或类似映射
    """
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            id2label = config.get('id2label', {})
            if id2label:
                return id2label
        except Exception:
            pass

    # 默认映射
    return {_BDCFG.get('human_label_id', 0): 'Human',
            _BDCFG.get('ai_label_id', 1): 'AI'}


def bert_detect_score(text, max_length=None):
    """用 BERT ONNX 模型检测文本的 AI 程度。

    Args:
        text: 待检测的中文文本
        max_length: 最大 token 长度，默认从配置读取（默认 128）

    Returns:
        float or None: 0-100 的 AI 概率分数，模型不可用时返回 None
    """
    if not _init():
        return None

    if max_length is None:
        max_length = _BDCFG.get('max_length', 128)

    try:
        # 温度校准参数（来自模型训练时的校准）
        temperature = _BDCFG.get('temperature', 0.8165)

        # Tokenize
        encoded = _tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )

        # 构造 ONNX 输入
        input_names = {inp.name for inp in _onnx_session.get_inputs()}
        feeds = {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64),
        }
        if 'token_type_ids' in input_names:
            feeds['token_type_ids'] = encoded.get(
                'token_type_ids',
                np.zeros_like(encoded['input_ids'])
            ).astype(np.int64)

        # 推理
        outputs = _onnx_session.run(None, feeds)

        # 解析输出: SequenceClassification logits (1, 2)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0][0]  # (2,)
        else:
            logits = outputs[0]  # (2,)

        # 温度校准 + softmax
        logits_scaled = logits / temperature
        probs = np.exp(logits_scaled - np.max(logits_scaled))
        probs = probs / probs.sum()

        # 找到 AI 标签的概率
        ai_label_id = _BDCFG.get('ai_label_id', 1)
        ai_prob = probs[ai_label_id]

        # 转为 0-100 分数
        score = round(float(ai_prob) * 100, 1)
        return score

    except Exception as e:
        logger.debug("BERT 检测推理异常: %s", e)
        return None


def bert_detect_batch(texts, max_length=None, batch_size=8):
    """批量检测多段文本。

    Args:
        texts: 文本列表
        max_length: 最大 token 长度
        batch_size: 批量大小

    Returns:
        list[float or None]: 每段文本的 AI 概率分数
    """
    if not _init():
        return [None] * len(texts)

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            results.append(bert_detect_score(text, max_length))
    return results


# ─── 便捷函数：供 detect_cn.py 直接调用 ───

def get_bert_score(text):
    """供 detect_cn.py 调用的统一接口。

    Returns:
        dict or None: {'score': float, 'source': 'bert_detector'}
                       模型不可用时返回 None
    """
    score = bert_detect_score(text)
    if score is None:
        return None

    return {
        'score': score,
        'source': 'bert_detector',
    }

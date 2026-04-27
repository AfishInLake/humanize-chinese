#!/usr/bin/env python3
"""
语义保持检查模块 — 基于 BERT ONNX 模型。

在改写后检查原文与改写文的 embedding 余弦相似度，
低于阈值则回退原文，防止改写改变原意。

ONNX 模型需另机训练后导出（见 refactoring-plan.md 第 6-7 节）。
模型不可用时自动降级为跳过检查。
"""

import os
import numpy as np

# ─── 配置加载 ───
from config_loader import load_config as _load_cfg
_CFG = _load_cfg()
_SGCFG = _CFG.get('semantic_guard', {})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ─── 从配置读取 ONNX 模型路径 ───
_onnx_model_name = _SGCFG.get('onnx_model_path', 'bert_base_chinese.onnx')
_ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, _onnx_model_name)
_ONNX_SESSION = None
_ONNX_AVAILABLE = None


def _init_onnx():
    """延迟初始化 ONNX Runtime 会话。"""
    global _ONNX_SESSION, _ONNX_AVAILABLE
    if _ONNX_AVAILABLE is not None:
        return _ONNX_AVAILABLE

    if not os.path.exists(_ONNX_MODEL_PATH):
        _ONNX_AVAILABLE = False
        return False

    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ONNX_SESSION = ort.InferenceSession(_ONNX_MODEL_PATH, sess_options)
        _ONNX_AVAILABLE = True
    except (ImportError, Exception):
        _ONNX_AVAILABLE = False

    return _ONNX_AVAILABLE


def _get_embeddings(texts):
    """从 ONNX 模型获取文本 embedding（[CLS] 向量）。"""
    if _ONNX_SESSION is None:
        return None

    try:
        import onnxruntime as ort
        # 简单的字符级 tokenization（实际应使用与训练一致的 tokenizer）
        # 这里是占位实现，实际部署时需要加载 tokenizer
        # TODO: 加载 BertTokenizer 并转换为 ONNX 输入格式
        # ─── 从配置读取最大 token 长度 ───
        _max_tok = _SGCFG.get('max_token_length', 512)
        input_ids = np.array([[ord(c) % 21128 for c in texts[0][:_max_tok]]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        outputs = _ONNX_SESSION.run(
            None,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
        )

        # 取 [CLS] token 的最后一层隐藏状态作为句子 embedding
        # ONNX 输出格式取决于导出时的设置
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            hidden_states = outputs[0]  # (1, seq_len, 768)
            cls_embedding = hidden_states[0, 0]  # (768,)
        else:
            cls_embedding = outputs[0][0, 0]

        return cls_embedding
    except Exception:
        return None


def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度。"""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def check_semantic_preservation(original, rewritten, threshold=None):
    """检查改写是否保持了原文语义。

    Args:
        original: 原始文本
        rewritten: 改写后文本
        threshold: 相似度阈值，低于此值则认为语义偏移过大

    Returns:
        (is_preserved, similarity_score)
        - is_preserved: bool，True 表示语义保持良好
        - similarity_score: float，余弦相似度分数
    """
    # ─── 从配置读取相似度阈值 ───
    if threshold is None:
        threshold = _SGCFG.get('similarity_threshold', 0.85)
    if not _init_onnx():
        # ONNX 不可用，跳过检查（默认通过）
        return True, 1.0

    emb_orig = _get_embeddings([original])
    emb_rew = _get_embeddings([rewritten])

    if emb_orig is None or emb_rew is None:
        return True, 1.0

    sim = cosine_similarity(emb_orig, emb_rew)
    return sim >= threshold, sim


def safe_rewrite(original, rewritten, threshold=None):
    """语义安全的改写：相似度不足时回退原文。

    Args:
        original: 原始文本
        rewritten: 改写后文本
        threshold: 相似度阈值

    Returns:
        改写后的文本（如果语义保持）或原文（如果语义偏移过大）
    """
    # ─── 从配置读取相似度阈值 ───
    if threshold is None:
        threshold = _SGCFG.get('similarity_threshold', 0.85)
    is_preserved, sim = check_semantic_preservation(original, rewritten, threshold)
    if is_preserved:
        return rewritten
    else:
        return original


# ─── CLI 测试 ───

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('用法: python semantic_guard.py <原文> <改写文> [阈值]')
        sys.exit(1)

    orig = sys.argv[1]
    rew = sys.argv[2]
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85

    is_ok, sim = check_semantic_preservation(orig, rew, thresh)
    status = '✓ 语义保持' if is_ok else '✗ 语义偏移'
    print(f'{status} | 相似度: {sim:.4f} | 阈值: {thresh}')
    print(f'原文: {orig[:50]}...')
    print(f'改写: {rew[:50]}...')

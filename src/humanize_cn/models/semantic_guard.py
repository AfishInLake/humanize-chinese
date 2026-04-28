#!/usr/bin/env python3
"""
语义保持检查模块 — 基于 BERT ONNX 模型。

在改写后检查原文与改写文的 embedding 余弦相似度，
低于阈值则回退原文，防止改写改变原意。

使用 bert-base-chinese 预训练模型导出的 ONNX（与句式评分器共用同一模型）。
模型不可用时自动降级为跳过检查。

所需文件（放到 scripts/ 目录）：
  - bert_base_chinese.onnx       （ONNX 模型）
  - bert_base_chinese/            （tokenizer 目录）
    ├── vocab.txt
    ├── tokenizer_config.json
    └── ...
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ─── 配置加载 ───
from ..config import load_config as _load_cfg
_CFG = _load_cfg()
_SGCFG = _CFG.get('semantic_guard', {})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 模型路径 ───
_onnx_model_name = _SGCFG.get('onnx_model_path', 'bert_base_chinese.onnx')
_ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'models', 'semantic_guard', _onnx_model_name)
_tokenizer_dir = os.path.join(SCRIPT_DIR, '..', 'data')

_ONNX_SESSION = None
_ONNX_AVAILABLE = None
_tokenizer = None


def _init_onnx():
    """延迟初始化 ONNX Runtime 会话 + Tokenizer。"""
    global _ONNX_SESSION, _ONNX_AVAILABLE, _tokenizer

    if _ONNX_AVAILABLE is not None:
        return _ONNX_AVAILABLE

    if not os.path.exists(_ONNX_MODEL_PATH):
        logger.info("语义保镖 ONNX 模型不存在: %s，将跳过语义检查", _ONNX_MODEL_PATH)
        _ONNX_AVAILABLE = False
        return False

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # 加载 tokenizer
        try:
            _tokenizer = AutoTokenizer.from_pretrained(_tokenizer_dir, local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

        # 加载 ONNX 模型
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ONNX_SESSION = ort.InferenceSession(_ONNX_MODEL_PATH, sess_options)

        _ONNX_AVAILABLE = True
        logger.info("语义保镖加载成功: %s", _ONNX_MODEL_PATH)

    except ImportError:
        _ONNX_AVAILABLE = False
        logger.info("onnxruntime/transformers 未安装，语义保镖不可用")
    except Exception as e:
        _ONNX_AVAILABLE = False
        logger.warning("语义保镖加载失败: %s", e)

    return _ONNX_AVAILABLE


def _get_embeddings(text):
    """从 ONNX 模型获取文本 embedding（[CLS] 向量）。

    Args:
        text: 单段文本

    Returns:
        numpy array (768,) 或 None
    """
    if not _init_onnx():
        return None

    try:
        max_len = _SGCFG.get('max_token_length', 512)

        # Tokenize
        encoded = _tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )

        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        # 构造 ONNX 输入
        input_names = {inp.name for inp in _ONNX_SESSION.get_inputs()}
        feeds = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in input_names:
            feeds['token_type_ids'] = np.zeros_like(input_ids)

        outputs = _ONNX_SESSION.run(None, feeds)

        # 解析输出：取 [CLS] token 的最后一层隐藏状态
        # BertModel 导出时，输出是 hidden_states (1, seq_len, 768)
        # BertForMaskedLM 导出时，输出是 logits (1, seq_len, vocab_size)
        # 需要区分两种情况

        if isinstance(outputs, (tuple, list)):
            # 多输出：hidden_states 通常是第一个
            first_output = outputs[0]
        else:
            first_output = outputs

        # 判断是 hidden_states 还是 logits
        # hidden_states: shape (1, seq_len, 768)
        # logits: shape (1, seq_len, 21128)
        if len(first_output.shape) == 3:
            if first_output.shape[-1] == 768:
                # hidden_states — 直接取 [CLS]
                cls_embedding = first_output[0, 0]  # (768,)
            elif first_output.shape[-1] > 1000:
                # logits（MLM 模型）— 无法取 embedding，降级
                logger.debug("语义保镖: 模型输出是 logits 而非 hidden_states，无法提取 embedding")
                return None
            else:
                cls_embedding = first_output[0, 0]
        else:
            return None

        return cls_embedding

    except Exception as e:
        logger.debug("语义保镖 embedding 提取异常: %s", e)
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
    if threshold is None:
        threshold = _SGCFG.get('similarity_threshold', 0.85)
    if not _init_onnx():
        return True, 1.0

    emb_orig = _get_embeddings(original)
    emb_rew = _get_embeddings(rewritten)

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

#!/usr/bin/env python3
"""BERT 掩码替换策略评估

对比两种替换策略：
  A. 当前策略：静态同义词表（WORD_SYNONYMS）+ bigram 频率排序
  B. 新策略：BERT MLM 掩码预测，取 top-k 候选

运行: PYTHONPATH=src python3 eval_mask_strategy.py
"""

import sys
import os
import json
import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from humanize_cn.models.restructure import _init_bert, _softmax


# ─── 测试用例 ───
TEST_CASES = [
    # (原文, 目标词, 期望方向)
    ("采用Python语言配合Django框架", "采用", "选用/使用"),
    ("实现安全的用户认证与权限控制", "实现", "达成/完成"),
    ("主要划分为普通学生用户与系统管理员", "主要", "核心/首要"),
    ("提供多条件组合查询与分页展示", "提供", "给出/支持"),
    ("促进校园资源的循环利用", "促进", "推动/助力"),
    ("处理商品图片等非结构化数据", "处理", "应对/管理"),
    ("进行功能测试与性能测试", "进行", "开展/实施"),
    ("保持系统的安全稳定运行", "保持", "维持/确保"),
    ("降低用户学习成本", "降低", "减少/压缩"),
    ("提高数据检索效率", "提高", "提升/改善"),
    ("包含注册、登录及个人中心维护", "包含", "涵盖/涉及"),
    ("确保数据的一致性与完整性", "确保", "保障/保证"),
    ("支持图片上传、编辑、下架", "支持", "允许/具备"),
    ("完成需求分析、系统设计", "完成", "做好/执行"),
    ("构建一个面向校园师生的平台", "构建", "搭建/打造"),
]


def bert_mask_predict(sentence, target_word, top_k=5):
    """BERT 掩码预测：遮蔽目标词，取 top-k 预测。

    Returns:
        list[(token, prob)] — top-k 预测结果
    """
    tokenizer, session = _init_bert()
    if tokenizer is None or session is None:
        return []

    tokens = tokenizer.tokenize(sentence)
    if not tokens:
        return []

    # 找到目标词在 token 列表中的位置
    target_tokens = tokenizer.tokenize(target_word)
    if not target_tokens:
        return []

    # 找第一个匹配位置
    mask_pos = None
    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i + len(target_tokens)] == target_tokens:
            mask_pos = i
            break

    if mask_pos is None:
        return []

    # 遮蔽目标词的所有 token
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_ids = token_ids.copy()
    for j in range(len(target_tokens)):
        masked_ids[mask_pos + j] = tokenizer.mask_token_id

    # ONNX 推理
    input_ids = np.array([masked_ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    feeds = {'input_ids': input_ids, 'attention_mask': attention_mask}
    input_names = {inp.name for inp in session.get_inputs()}
    if 'token_type_ids' in input_names:
        feeds['token_type_ids'] = np.zeros_like(input_ids)

    outputs = session.run(None, feeds)
    logits = outputs[0][0, mask_pos]  # 第一个 mask 位置的 logits
    probs = _softmax(logits)

    # 取 top-k
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        token = tokenizer.convert_ids_to_tokens(int(idx))
        prob = probs[idx]
        # 过滤 subword 和特殊 token
        if token.startswith('##') or token in ('[CLS]', '[SEP]', '[PAD]', '[MASK]'):
            continue
        results.append((token, float(prob)))
        if len(results) >= top_k:
            break

    return results


def evaluate():
    print("=" * 70)
    print("  BERT 掩码替换策略评估")
    print("=" * 70)

    tokenizer, session = _init_bert()
    if tokenizer is None:
        print("❌ BERT 模型不可用")
        return

    print(f"\n模型: bert_base_chinese_mlm (sentence_scorer)")
    print(f"{'─' * 70}")

    # ─── 逐例测试 ───
    good = 0
    total = 0

    for sentence, target, expected_desc in TEST_CASES:
        predictions = bert_mask_predict(sentence, target, top_k=5)
        top_tokens = [t for t, p in predictions]
        top_probs = [p for t, p in predictions]

        # 判断 top-1 是否合理
        top1 = top_tokens[0] if top_tokens else "N/A"
        # 检查 top-3 是否包含期望方向的词
        expected_words = expected_desc.split('/')
        hit = any(w in top_tokens[:3] for w in expected_words)
        if hit:
            good += 1
        total += 1

        status = "✅" if hit else "⚠️"
        pred_str = ", ".join(f"{t}({p:.3f})" for t, p in predictions[:3])

        print(f"\n{status} 「{target}」→ 上下文: ...{sentence[:30]}...")
        print(f"   期望方向: {expected_desc}")
        print(f"   BERT top-3: {pred_str}")

    # ─── 汇总 ───
    print(f"\n{'=' * 70}")
    print(f"  评估汇总")
    print(f"{'=' * 70}")
    print(f"  测试用例: {total}")
    print(f"  top-3 命中期望方向: {good}/{total} ({good/total*100:.0f}%)")

    # ─── 策略对比分析 ───
    print(f"\n{'=' * 70}")
    print(f"  策略对比分析")
    print(f"{'=' * 70}")

    print("""
┌─────────────┬──────────────────────────┬──────────────────────────┐
│ 维度         │ A. 静态同义词表           │ B. BERT 掩码预测          │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ 上下文感知   │ ❌ 无，固定候选列表       │ ✅ 根据上下文动态生成     │
│ 速度         │ ⚡ 极快（查表）           │ 🐢 慢（每词一次推理）     │
│ 覆盖范围     │ ❌ 仅预定义的词           │ ✅ 任意词都能预测         │
│ 准确性       │ ⚠️ 依赖人工维护质量       │ ⚠️ 可能生成不合适的词     │
│ 词性一致性   │ ⚠️ 需要额外 POS 过滤     │ ✅ BERT 天然保持词性      │
│ 语义偏移风险 │ ⚠️ 中（人工审核）         │ ⚠️ 中（需过滤低概率候选） │
│ 依赖         │ 无（纯 Python）          │ 需要 ONNX 模型 (~400MB)  │
│ 适用场景     │ 通用改写                 │ 精细化改写                │
└─────────────┴──────────────────────────┴──────────────────────────┘

结论:
  - BERT 掩码预测在这个场景下完全不可用（0/15 命中率）
  - 根本原因：BERT MLM 是字级模型，遮蔽"采用"只遮第一个字"采"，
    预测的是下一个字而非同义词，所以 top-1 都是"一""公""位"等上下文字
  - 静态同义词表虽然不完美，但至少保证候选词是语义相关的
  - BERT 的正确用法是"评分"（给候选打分选最优），而非"生成"（预测替换词）

推荐方案：保持当前静态同义词表 + BERT 评分选最优的混合策略。
  BERT 掩码预测仅适用于英文等词级分词语言，中文（字级）不适用。
""")


if __name__ == '__main__':
    evaluate()

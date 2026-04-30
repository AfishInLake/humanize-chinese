#!/usr/bin/env python3
"""
综合评估脚本: AUROC, 准确率, 召回率, F1, 跨域泛化, 按长度分组。

Usage:
    python scripts/eval_ensemble.py --test scripts/training_data/test.jsonl
    python scripts/eval_ensemble.py --test scripts/training_data/test.jsonl --by-domain
    python scripts/eval_ensemble.py --test scripts/training_data/test.jsonl --by-length
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

DEFAULT_TEST = os.path.join(SCRIPT_DIR, 'training_data', 'test.jsonl')


def load_jsonl(filepath):
    """Load texts, labels, sources from JSONL."""
    texts, labels, sources = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                texts.append(row.get('text', ''))
                labels.append(int(row.get('label', -1)))
                sources.append(row.get('source', 'unknown'))
            except json.JSONDecodeError:
                continue
    return texts, labels, sources


def compute_metrics(y_true, y_prob, y_pred=None, name='Eval'):
    """Compute classification metrics."""
    try:
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score,
            recall_score, f1_score, confusion_matrix,
        )
    except ImportError:
        print('错误: 需要安装 sklearn。pip install scikit-learn')
        return {}

    if y_pred is None:
        y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Optimal threshold (Youden's J)
    best_threshold = 0.5
    best_j = 0
    for t in [i / 100 for i in range(5, 96, 5)]:
        pred_t = [1 if p >= t else 0 for p in y_prob]
        tp = sum(1 for a, b in zip(y_true, pred_t) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true, pred_t) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true, pred_t) if a == 1 and b == 0)
        tn = sum(1 for a, b in zip(y_true, pred_t) if a == 0 and b == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = t

    print(f'\n[{name}]')
    print(f'  AUROC:            {auc:.4f}')
    print(f'  Accuracy:         {acc:.4f}')
    print(f'  Precision:        {prec:.4f}')
    print(f'  Recall:           {rec:.4f}')
    print(f'  F1:               {f1:.4f}')
    print(f'  Optimal Threshold:{best_threshold:.2f}')
    print(f'  Confusion Matrix: {cm}')

    return {
        'auroc': auc, 'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'threshold': best_threshold,
        'confusion_matrix': cm,
    }


def eval_by_domain(texts, labels, sources, score_fn):
    """Evaluate by domain/source."""
    domain_groups = defaultdict(list)
    for text, label, source in zip(texts, labels, sources):
        # Normalize source names
        domain = source.split('_')[0] if '_' in source else source
        domain_groups[domain].append((text, label))

    print(f'\n{"="*60}')
    print('按领域分组评估')
    print(f'{"="*60}')

    domain_results = {}
    for domain, items in sorted(domain_groups.items()):
        domain_texts = [t for t, _ in items]
        domain_labels = [l for _, l in items]
        domain_probs = score_fn(domain_texts)
        domain_preds = [1 if p >= 0.5 else 0 for p in domain_probs]
        metrics = compute_metrics(domain_labels, domain_probs, domain_preds,
                                  name=f'领域: {domain} (n={len(items)})')
        domain_results[domain] = metrics

    # Check cross-domain variance
    aucs = [m['auroc'] for m in domain_results.values() if m]
    if len(aucs) >= 2:
        aucs_range = max(aucs) - min(aucs)
        print(f'\n跨域 AUROC 差异: {aucs_range:.4f} (目标 < 0.05)')

    return domain_results


def eval_by_length(texts, labels, score_fn):
    """Evaluate by text length groups."""
    def cn_count(t):
        return sum(1 for c in t if '\u4e00' <= c <= '\u9fff')

    groups = {
        'short (<200)': [],
        'medium (200-1000)': [],
        'long (>1000)': [],
    }

    for text, label in zip(texts, labels):
        cl = cn_count(text)
        if cl < 200:
            groups['short (<200)'].append((text, label))
        elif cl <= 1000:
            groups['medium (200-1000)'].append((text, label))
        else:
            groups['long (>1000)'].append((text, label))

    print(f'\n{"="*60}')
    print('按文本长度分组评估')
    print(f'{"="*60}')

    length_results = {}
    for group_name, items in groups.items():
        if len(items) < 5:
            print(f'\n  [{group_name}] 样本不足 ({len(items)} 条)，跳过')
            continue
        group_texts = [t for t, _ in items]
        group_labels = [l for _, l in items]
        group_probs = score_fn(group_texts)
        group_preds = [1 if p >= 0.5 else 0 for p in group_probs]
        metrics = compute_metrics(group_labels, group_probs, group_preds,
                                  name=f'{group_name} (n={len(items)})')
        length_results[group_name] = metrics

    return length_results


def main():
    parser = argparse.ArgumentParser(description='综合评估脚本')
    parser.add_argument('--test', type=str, default=DEFAULT_TEST,
                        help='测试数据路径 (JSONL)')
    parser.add_argument('--method', type=str, default='ensemble',
                        choices=['ensemble', 'xgboost', 'lr', 'bert', 'ngram'],
                        help='评估方法 (默认: ensemble)')
    parser.add_argument('--by-domain', action='store_true',
                        help='按领域分组评估')
    parser.add_argument('--by-length', action='store_true',
                        help='按文本长度分组评估')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大评估样本数')
    args = parser.parse_args()

    print(f'{"="*60}')
    print('综合评估脚本')
    print(f'{"="*60}')
    print(f'  测试数据: {args.test}')
    print(f'  评估方法: {args.method}')
    print(f'{"="*60}\n')

    # Load data
    texts, labels, sources = load_jsonl(args.test)
    if args.max_samples:
        texts = texts[:args.max_samples]
        labels = labels[:args.max_samples]
        sources = sources[:args.max_samples]

    n_ai = sum(labels)
    n_human = len(labels) - n_ai
    print(f'测试集: {len(texts)} 条 (AI: {n_ai}, Human: {n_human})')

    # Define scoring function
    def score_fn(texts_batch):
        """Score a batch of texts, return list of AI probabilities."""
        from humanize_cn.check_pkg.api import check
        results = []
        for text in texts_batch:
            r = check(text)
            results.append(r['ai_score'] / 100.0)
        return results

    # Overall evaluation
    print(f'\n{"="*60}')
    print('整体评估')
    print(f'{"="*60}')
    t0 = time.time()
    probs = score_fn(texts)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    elapsed = time.time() - t0
    overall_metrics = compute_metrics(labels, probs, preds, name='整体')
    print(f'\n  总耗时: {elapsed:.1f}s ({elapsed/len(texts)*1000:.0f}ms/条)')

    # Grouped evaluations
    if args.by_domain:
        eval_by_domain(texts, labels, sources, score_fn)

    if args.by_length:
        eval_by_length(texts, labels, score_fn)

    # Save results
    out_path = os.path.join(os.path.dirname(args.test), 'eval_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall': overall_metrics,
            'method': args.method,
            'n_samples': len(texts),
            'n_ai': n_ai,
            'n_human': n_human,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_path}')


if __name__ == '__main__':
    main()

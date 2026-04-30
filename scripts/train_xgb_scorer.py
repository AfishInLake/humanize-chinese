#!/usr/bin/env python3
"""
Train XGBoost / Random Forest ensemble scorer on extended feature set.

Uses the 37-dimension feature vector from ngram.extract_feature_vector()
to train a tree-based classifier for AI text detection.

Usage:
    python scripts/train_xgb_scorer.py
    python scripts/train_xgb_scorer.py --classifier xgboost --n-estimators 300
    python scripts/train_xgb_scorer.py --classifier rf --n-estimators 500
    python scripts/train_xgb_scorer.py --data scripts/training_data/train.jsonl \
        --val scripts/training_data/val.jsonl

Output: scripts/xgb_model_cn.json (model + scaler metadata)
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from statistics import mean, stdev

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

DEFAULT_TRAIN = os.path.join(REPO_ROOT, 'output', 'train.jsonl')
DEFAULT_VAL = os.path.join(REPO_ROOT, 'output', 'val.jsonl')
DEFAULT_OUT = os.path.join(REPO_ROOT, 'output', 'xgb_model_cn.json')


def load_jsonl(filepath, max_samples=None):
    """Load texts and labels from JSONL file."""
    texts, labels, sources = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text = row.get('text', '')
                label = row.get('label')
                source = row.get('source', 'unknown')
                if text and label is not None:
                    texts.append(text)
                    labels.append(int(label))
                    sources.append(source)
            except json.JSONDecodeError:
                continue
            if max_samples and len(texts) >= max_samples:
                break
    return texts, labels, sources


def extract_features_batch(texts, show_progress=True):
    """Extract 37-dim feature vectors for a list of texts.

    Imports ngram module from src/humanize_cn/models/.
    """
    from humanize_cn.models.ngram import extract_feature_vector

    vectors = []
    for i, text in enumerate(texts):
        vec, names = extract_feature_vector(text, version='extended')
        vectors.append(vec)
        if show_progress and (i + 1) % 100 == 0:
            print(f'  特征提取: {i+1}/{len(texts)}')
    return vectors


def standardize_fit(X_train):
    """Compute mean and scale for standardization."""
    n_feat = len(X_train[0])
    means = [mean(x[f] for x in X_train) for f in range(n_feat)]
    scales = []
    for f in range(n_feat):
        s = stdev([x[f] for x in X_train]) or 1.0
        scales.append(s)
    return means, scales


def standardize_apply(X, means, scales):
    """Apply standardization."""
    return [[(x[f] - means[f]) / (scales[f] if scales[f] else 1.0)
             for f in range(len(means))] for x in X]


def train_xgboost(X_train, y_train, X_val, y_val, args):
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        print('错误: 未安装 xgboost。请运行: pip install xgboost')
        sys.exit(1)

    print(f'[XGBoost] 训练参数:')
    print(f'  n_estimators={args.n_estimators}, max_depth={args.max_depth}')
    print(f'  learning_rate={args.learning_rate}, subsample={args.subsample}')

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        eval_metric='logloss',
        early_stopping_rounds=20,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_rf(X_train, y_train, X_val, y_val, args):
    """Train Random Forest classifier."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print('错误: 未安装 sklearn。请运行: pip install scikit-learn')
        sys.exit(1)

    print(f'[RF] 训练参数:')
    print(f'  n_estimators={args.n_estimators}, max_depth={args.max_depth}')

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, X, y, name='Eval'):
    """Evaluate model and print metrics."""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score, classification_report,
    )

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print(f'\n[{name}] 结果:')
    print(f'  AUROC:      {auc:.4f}')
    print(f'  Accuracy:   {acc:.4f}')
    print(f'  Precision:  {prec:.4f}')
    print(f'  Recall:     {rec:.4f}')
    print(f'  F1:         {f1:.4f}')

    return {'auroc': auc, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1}


def get_feature_importance(model, feature_names):
    """Get feature importance from model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        # XGBoost
        booster = model.get_booster()
        scores = booster.get_score(importance_type='gain')
        # Map f0, f1, ... to feature names
        importances = [scores.get(f'f{i}', 0.0) for i in range(len(feature_names))]
    else:
        return []

    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description='XGBoost/RF 集成分类器训练脚本')
    parser.add_argument('--data', type=str, default=DEFAULT_TRAIN,
                        help='训练数据路径 (JSONL)')
    parser.add_argument('--val', type=str, default=DEFAULT_VAL,
                        help='验证数据路径 (JSONL)')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT,
                        help='模型输出路径')
    parser.add_argument('--classifier', type=str, default='xgboost',
                        choices=['xgboost', 'rf'],
                        help='分类器类型 (默认: xgboost)')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='树的数量 (默认: 300)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='最大深度 (默认: 6)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='学习率，仅 XGBoost (默认: 0.1)')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='子采样比例，仅 XGBoost (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大训练样本数 (默认: 全部)')
    args = parser.parse_args()

    print(f'{"="*60}')
    print('XGBoost/RF 集成分类器训练脚本')
    print(f'{"="*60}')
    print(f'  训练数据:   {args.data}')
    print(f'  验证数据:   {args.val}')
    print(f'  分类器:     {args.classifier}')
    print(f'  输出:       {args.out}')
    print(f'{"="*60}\n')

    # 1. 加载数据
    print('[步骤 1/5] 加载数据...')
    train_texts, train_labels, train_sources = load_jsonl(
        args.data, max_samples=args.max_samples)
    val_texts, val_labels, val_sources = load_jsonl(args.val)

    n_train_ai = sum(train_labels)
    n_train_human = len(train_labels) - n_train_ai
    n_val_ai = sum(val_labels)
    n_val_human = len(val_labels) - n_val_ai
    print(f'  训练集: {len(train_texts)} 条 (AI: {n_train_ai}, Human: {n_train_human})')
    print(f'  验证集: {len(val_texts)} 条 (AI: {n_val_ai}, Human: {n_val_human})')

    # 2. 提取特征
    print(f'\n[步骤 2/5] 提取特征 (37 维)...')
    t0 = time.time()
    X_train_raw = extract_features_batch(train_texts)
    X_val_raw = extract_features_batch(val_texts)
    t1 = time.time()
    print(f'  特征提取完成，耗时 {t1-t0:.1f}s')

    # 3. 标准化
    print(f'\n[步骤 3/5] 标准化...')
    feat_means, feat_scales = standardize_fit(X_train_raw)
    X_train = standardize_apply(X_train_raw, feat_means, feat_scales)
    X_val = standardize_apply(X_val_raw, feat_means, feat_scales)

    # 4. 训练
    print(f'\n[步骤 4/5] 训练 {args.classifier.upper()}...')
    if args.classifier == 'xgboost':
        model = train_xgboost(X_train, train_labels, X_val, val_labels, args)
    else:
        model = train_rf(X_train, train_labels, X_val, val_labels, args)

    # 5. 评估
    print(f'\n[步骤 5/5] 评估...')
    train_metrics = evaluate(model, X_train, train_labels, '训练集')
    val_metrics = evaluate(model, X_val, val_labels, '验证集')

    # 特征重要性
    from humanize_cn.models.ngram import EXTENDED_FEATURE_NAMES
    importance = get_feature_importance(model, EXTENDED_FEATURE_NAMES)
    if importance:
        print(f'\n特征重要性 (Top 15):')
        for name, imp in importance[:15]:
            print(f'  {name:<30} {imp:.4f}')

    # 保存模型
    print(f'\n[保存] 模型到 {args.out}')
    if args.classifier == 'xgboost':
        model.save_model(args.out)
    else:
        # RF: save as joblib or pickle
        import joblib
        joblib.dump(model, args.out.replace('.json', '.joblib'))

    # 保存元信息
    meta = {
        'version': '1.0.0',
        'classifier': args.classifier,
        'trained_at': datetime.now().isoformat(),
        'n_features': len(EXTENDED_FEATURE_NAMES),
        'feature_names': list(EXTENDED_FEATURE_NAMES),
        'mean': feat_means,
        'scale': feat_scales,
        'n_train': len(train_texts),
        'n_val': len(val_texts),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'params': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'subsample': args.subsample,
            'seed': args.seed,
        },
    }
    meta_path = args.out.replace('.json', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'[保存] 元信息到 {meta_path}')

    print(f'\n{"="*60}')
    print('训练完成!')
    print(f'{"="*60}')
    print(f'  验证集 AUROC: {val_metrics["auroc"]:.4f}')
    print(f'  模型文件: {args.out}')
    print(f'  元信息:   {meta_path}')


if __name__ == '__main__':
    main()

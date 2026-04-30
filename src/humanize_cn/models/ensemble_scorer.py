#!/usr/bin/env python3
"""
Ensemble scorer: XGBoost + LR fallback for AI text detection.

Loads XGBoost model (JSON format) and scores text with the extended 37-dim
feature vector. Falls back to LR scoring when XGBoost is unavailable.

Usage:
    from humanize_cn.models.ensemble_scorer import ensemble_score
    result = ensemble_score("一段文本")
    # result = {'p_ai': float, 'score': int, 'method': 'xgboost'|'lr'|None}
"""

import os
import json
from loguru import logger
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Configuration ───
from ..config import load_config as _load_cfg
_CFG = _load_cfg()
_ECFG = _CFG.get('ensemble', {})

# ─── Global state (lazy init) ───
_xgb_model = None
_xgb_meta = None
_xgb_available = None  # None=not checked, True/False=checked


def _init_xgb():
    """Lazy-load XGBoost model and metadata."""
    global _xgb_model, _xgb_meta, _xgb_available

    if _xgb_available is not None:
        return _xgb_available

    model_path = _ECFG.get('model_path',
                           'models/ensemble/xgb_model_cn.json')
    # 如果是相对路径，基于 data/ 目录解析
    if not os.path.isabs(model_path):
        data_dir = os.path.join(SCRIPT_DIR, '..', 'data')
        model_path = os.path.join(data_dir, model_path)
    model_path = os.path.normpath(model_path)
    meta_path = model_path.replace('.json', '_meta.json')

    if not os.path.exists(model_path):
        logger.info("XGBoost model not found: {}", model_path)
        _xgb_available = False
        return False

    try:
        import xgboost as xgb

        _xgb_model = xgb.XGBClassifier()
        _xgb_model.load_model(model_path)

        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                _xgb_meta = json.load(f)
        else:
            _xgb_meta = {}

        _xgb_available = True
        logger.info("XGBoost ensemble model loaded: {}", model_path)

    except ImportError:
        _xgb_available = False
        logger.info("xgboost not installed, ensemble scorer disabled")
    except Exception as e:
        _xgb_available = False
        logger.warning("XGBoost model load failed: {}", e)

    return _xgb_available


def _xgb_predict(vec, means, scales):
    """Run XGBoost prediction on a standardized feature vector.

    Args:
        vec: raw feature vector (list of floats)
        means: standardization means
        scales: standardization scales

    Returns:
        float: AI probability (0-1)
    """
    # Standardize
    standardized = [(vec[i] - means[i]) / (scales[i] if scales[i] else 1.0)
                    for i in range(min(len(vec), len(means)))]

    # Pad or truncate to expected length
    expected_len = len(means)
    if len(standardized) < expected_len:
        standardized.extend([0.0] * (expected_len - len(standardized)))
    else:
        standardized = standardized[:expected_len]

    # Predict
    import numpy as np
    X = np.array([standardized], dtype=np.float32)
    proba = _xgb_model.predict_proba(X)[0]
    return float(proba[1])  # P(AI)


def _try_xgb_score(vec):
    """Try XGBoost scoring. Returns dict or None."""
    if not _init_xgb():
        return None

    try:
        means = _xgb_meta.get('mean', [])
        scales = _xgb_meta.get('scale', [])
        if not means or not scales:
            return None

        p_ai = _xgb_predict(vec, means, scales)
        score = round(100 * p_ai)

        return {
            'p_ai': p_ai,
            'score': score,
            'method': 'xgboost',
        }
    except Exception as e:
        logger.debug("XGBoost prediction failed: {}", e)
        return None


def ensemble_score(text, scene='auto'):
    """Score text using ensemble: XGBoost priority, LR fallback.

    Computes the extended 37-dim feature vector and runs through XGBoost.
    If XGBoost is unavailable, falls back to LR scoring.

    Args:
        text: input Chinese text
        scene: 'auto', 'general', 'academic', or 'novel' (for LR fallback)

    Returns:
        dict with:
          - p_ai: AI probability (0-1)
          - score: AI probability 0-100
          - method: 'xgboost' / 'lr' / None
          - features: feature vector (for debugging)
    """
    from .ngram import extract_feature_vector, compute_lr_score

    # Extract extended feature vector
    vec, names = extract_feature_vector(text, version='extended')

    # Try XGBoost
    xgb_result = _try_xgb_score(vec)
    if xgb_result is not None:
        xgb_result['features'] = dict(zip(names, vec))
        return xgb_result

    # Fallback to LR
    lr_result = compute_lr_score(text, scene=scene)
    if lr_result is not None:
        return {
            'p_ai': lr_result['p_ai'],
            'score': lr_result['score'],
            'method': 'lr',
            'features': lr_result.get('features', {}),
        }

    return None


def ensemble_score_batch(texts, scene='auto', batch_size=None):
    """Score multiple texts using ensemble.

    Args:
        texts: list of text strings
        scene: scene for LR fallback
        batch_size: if set, process in batches (for XGBoost efficiency)

    Returns:
        list of result dicts (same format as ensemble_score)
    """
    from .ngram import extract_feature_vector

    # Extract all feature vectors
    all_vecs = []
    all_names = None
    for text in texts:
        vec, names = extract_feature_vector(text, version='extended')
        all_vecs.append(vec)
        if all_names is None:
            all_names = names

    # Try XGBoost batch prediction
    if _init_xgb() and _xgb_meta:
        try:
            means = _xgb_meta.get('mean', [])
            scales = _xgb_meta.get('scale', [])
            if means and scales:
                import numpy as np

                # Standardize all vectors
                X = []
                for vec in all_vecs:
                    std = [(vec[i] - means[i]) / (scales[i] if scales[i] else 1.0)
                           for i in range(min(len(vec), len(means)))]
                    expected_len = len(means)
                    if len(std) < expected_len:
                        std.extend([0.0] * (expected_len - len(std)))
                    else:
                        std = std[:expected_len]
                    X.append(std)

                X = np.array(X, dtype=np.float32)
                probas = _xgb_model.predict_proba(X)[:, 1]

                results = []
                for i, p_ai in enumerate(probas):
                    results.append({
                        'p_ai': float(p_ai),
                        'score': round(100 * float(p_ai)),
                        'method': 'xgboost',
                        'features': dict(zip(all_names, all_vecs[i])),
                    })
                return results
        except Exception as e:
            logger.debug("XGBoost batch prediction failed: {}", e)

    # Fallback: individual scoring
    return [ensemble_score(text, scene=scene) for text in texts]

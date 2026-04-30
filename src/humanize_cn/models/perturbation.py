#!/usr/bin/env python3
"""
Perturbation-based detection features (Fast-DetectGPT inspired).

Core idea: perturb text by replacing words with CiLin synonyms and observe
how the model's scoring changes.
  - AI text: larger score change after perturbation (sits on "low curvature"
    regions of the probability surface)
  - Human text: smaller score change (sits on "high curvature" regions)

Complementary to the character-level compute_curvature() in ngram.py:
  - compute_curvature: character-level, uses ngram table for alternative lookup
  - perturbation: word-level, uses CiLin synonym dictionary for real replacement

All features gracefully degrade to 0.0 when dependencies are unavailable.
"""

import os
import re
from loguru import logger
from math import log2
from collections import Counter



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CILIN_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'cilin_synonyms.json')

# ─── Configuration ───
from ..config import load_config as _load_cfg
_CFG = _load_cfg()
_PCFG = _CFG.get('ensemble', {})

# ─── CiLin synonym cache ───
_CILIN_CACHE = None


def _load_cilin():
    """Lazy-load CiLin synonym dictionary."""
    global _CILIN_CACHE
    if _CILIN_CACHE is not None:
        return _CILIN_CACHE

    if not os.path.exists(_CILIN_FILE):
        _CILIN_CACHE = {}
        return _CILIN_CACHE

    try:
        import json
        with open(_CILIN_FILE, 'r', encoding='utf-8') as f:
            _CILIN_CACHE = json.load(f)
        logger.info("CiLin synonyms loaded: {} entries", len(_CILIN_CACHE))
    except Exception as e:
        logger.warning("CiLin synonyms load failed: {}", e)
        _CILIN_CACHE = {}

    return _CILIN_CACHE


def _get_synonyms(word: str) -> list:
    """Get synonyms for a word from CiLin dictionary.

    Returns list of synonym strings, or empty list if none found.
    """
    cilin = _load_cilin()
    if not cilin:
        return []

    # CiLin format: word -> [synonym1, synonym2, ...]
    synonyms = cilin.get(word, [])
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    return [s for s in synonyms if s != word and len(s) > 0]


def _tokenize_words(text: str) -> list:
    """Simple word tokenization using jieba. Falls back to character split."""
    try:
        import jieba
        words = list(jieba.cut(text))
        return [(w, i) for i, w in enumerate(words) if w.strip()]
    except ImportError:
        # Fallback: split by Chinese characters and punctuation
        tokens = re.findall(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+', text)
        result = []
        pos = 0
        for t in tokens:
            result.append((t, pos))
            pos += len(t)
        return result


def _compute_ngram_score(text: str) -> float:
    """Compute a simple ngram log-prob score for text.

    Uses the ngram module's compute_perplexity if available.
    Returns average log-prob (higher = more predictable = more AI-like).
    """
    try:
        from .ngram import compute_perplexity
        result = compute_perplexity(text)
        if result.get('char_count', 0) > 0:
            return result.get('avg_log_prob', 0.0)
    except Exception:
        pass
    return 0.0


def _perturb_text(text: str, words: list, replace_ratio: float = 0.1, seed: int = 42) -> str:
    """Create a perturbed version of text by replacing words with synonyms.

    Args:
        text: original text
        words: list of (word, position) tuples from _tokenize_words
        replace_ratio: fraction of words to replace (default 0.1)
        seed: random seed

    Returns:
        str: perturbed text
    """
    import random as _random
    rng = _random.Random(seed)

    # Filter to replaceable words (Chinese, length >= 2, has synonyms)
    replaceable = []
    for w, pos in words:
        if len(w) >= 2 and sum(1 for c in w if '\u4e00' <= c <= '\u9fff') >= 2:
            syns = _get_synonyms(w)
            if syns:
                replaceable.append((w, pos, syns))

    if not replaceable:
        return text

    # Sample words to replace
    n_replace = max(1, int(len(replaceable) * replace_ratio))
    to_replace = rng.sample(replaceable, min(n_replace, len(replaceable)))

    # Build perturbed text
    perturbed = text
    offset = 0
    for orig_word, pos, syns in to_replace:
        replacement = rng.choice(syns)
        # Find and replace first occurrence after offset
        idx = perturbed.find(orig_word, offset)
        if idx >= 0:
            perturbed = perturbed[:idx] + replacement + perturbed[idx + len(orig_word):]
            offset = idx + len(replacement)

    return perturbed


def compute_perturbation_features(text: str, n_perturbations: int = None,
                                   replace_ratio: float = None,
                                   seed: int = 42) -> dict:
    """Compute perturbation sensitivity features.

    For each perturbation:
    1. Replace ~10% of words with CiLin synonyms
    2. Compute ngram log-prob of original and perturbed text
    3. Record the difference

    Args:
        text: input text
        n_perturbations: number of perturbation trials (default from config, 5)
        replace_ratio: fraction of words to replace (default from config, 0.1)
        seed: random seed

    Returns:
        dict with:
          - perturb_sensitivity: mean absolute log-prob change
          - perturb_std: std of log-prob changes
          - perturb_direction: fraction of changes that increase log-prob
          - available: whether computation succeeded
    """
    if n_perturbations is None:
        n_perturbations = _PCFG.get('n_perturbations', 5)
    if replace_ratio is None:
        replace_ratio = _PCFG.get('replace_ratio', 0.1)

    # Check if perturbation is enabled
    if not _PCFG.get('perturbation_enabled', True):
        return {
            'perturb_sensitivity': 0.0, 'perturb_std': 0.0,
            'perturb_direction': 0.0, 'available': False,
        }

    # Check minimum text length
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cn_chars < 50:
        return {
            'perturb_sensitivity': 0.0, 'perturb_std': 0.0,
            'perturb_direction': 0.0, 'available': True,
        }

    # Tokenize
    words = _tokenize_words(text)
    if len(words) < 5:
        return {
            'perturb_sensitivity': 0.0, 'perturb_std': 0.0,
            'perturb_direction': 0.0, 'available': True,
        }

    # Compute original score
    original_score = _compute_ngram_score(text)

    # Run perturbations
    changes = []
    for i in range(n_perturbations):
        perturbed = _perturb_text(text, words, replace_ratio, seed=seed + i)
        if perturbed == text:
            continue  # no words were replaced
        perturbed_score = _compute_ngram_score(perturbed)
        change = perturbed_score - original_score
        changes.append(change)

    if not changes:
        return {
            'perturb_sensitivity': 0.0, 'perturb_std': 0.0,
            'perturb_direction': 0.0, 'available': True,
        }

    abs_changes = [abs(c) for c in changes]
    sensitivity = sum(abs_changes) / len(abs_changes)

    mean_change = sum(changes) / len(changes)
    variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
    std = variance ** 0.5

    # Direction: fraction of positive changes (perturbed text more predictable)
    direction = sum(1 for c in changes if c > 0) / len(changes)

    return {
        'perturb_sensitivity': sensitivity,
        'perturb_std': std,
        'perturb_direction': direction,
        'available': True,
    }


def compute_local_curvature_v2(text: str, n_positions: int = None,
                                seed: int = 42) -> dict:
    """Compute word-level local curvature using CiLin synonyms.

    For each sampled position:
    1. Find CiLin synonyms for the word
    2. Replace with each synonym and compute ngram log-prob
    3. curvature = log_p(original) - mean(log_p(alternatives))

    Complementary to character-level compute_curvature() in ngram.py.

    Args:
        text: input text
        n_positions: number of positions to evaluate (default from config, 20)
        seed: random seed

    Returns:
        dict with:
          - local_curv_v2: mean curvature across positions
          - n_positions: number of positions evaluated
          - available: whether computation succeeded
    """
    if n_positions is None:
        n_positions = _PCFG.get('curvature_n_positions', 20)

    if not _PCFG.get('perturbation_enabled', True):
        return {'local_curv_v2': 0.0, 'n_positions': 0, 'available': False}

    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cn_chars < 50:
        return {'local_curv_v2': 0.0, 'n_positions': 0, 'available': True}

    words = _tokenize_words(text)

    # Filter to words with synonyms
    candidates = []
    for w, pos in words:
        if len(w) >= 2 and sum(1 for c in w if '\u4e00' <= c <= '\u9fff') >= 2:
            syns = _get_synonyms(w)
            if syns:
                candidates.append((w, pos, syns))

    if not candidates:
        return {'local_curv_v2': 0.0, 'n_positions': 0, 'available': True}

    import random as _random
    rng = _random.Random(seed)

    if len(candidates) > n_positions:
        candidates = rng.sample(candidates, n_positions)

    curvatures = []
    for orig_word, pos, syns in candidates:
        # Compute original score
        original_score = _compute_ngram_score(text)

        # Compute scores with synonyms
        alt_scores = []
        for syn in syns[:5]:  # limit to 5 synonyms per word
            perturbed = text.replace(orig_word, syn, 1)
            if perturbed != text:
                alt_scores.append(_compute_ngram_score(perturbed))

        if alt_scores:
            mean_alt = sum(alt_scores) / len(alt_scores)
            curvature = original_score - mean_alt
            curvatures.append(curvature)

    if not curvatures:
        return {'local_curv_v2': 0.0, 'n_positions': 0, 'available': True}

    return {
        'local_curv_v2': sum(curvatures) / len(curvatures),
        'n_positions': len(curvatures),
        'available': True,
    }

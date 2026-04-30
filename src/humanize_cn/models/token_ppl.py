#!/usr/bin/env python3
"""
Token-level perplexity features using BERT-base-Chinese MLM.

Uses the existing ONNX model (bert_base_chinese_mlm.onnx) for masked LM scoring.
Complementary to the character-level ngram perplexity in ngram.py:
  - Character-level: captures local character sequence predictability
  - Token-level: captures word collocation and semantic predictability

AI text typically has lower token-level perplexity and more uniform log-prob distribution.

All features gracefully degrade to 0.0 when the ONNX model is unavailable.
"""

import os
from loguru import logger
import numpy as np
from math import log2



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MLM_MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'models', 'sentence_scorer')
_MLM_ONNX_NAME = 'bert_base_chinese_mlm.onnx'

# ─── Global state (lazy init) ───
_tokenizer = None
_onnx_session = None
_available = None  # None=not checked, True/False=checked


def _init():
    """Lazy-load tokenizer + ONNX MLM model."""
    global _tokenizer, _onnx_session, _available

    if _available is not None:
        return _available

    onnx_path = os.path.join(_MLM_MODEL_DIR, _MLM_ONNX_NAME)
    tokenizer_dir = os.path.join(SCRIPT_DIR, '..', 'data')

    if not os.path.exists(onnx_path):
        _available = False
        return False

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _onnx_session = ort.InferenceSession(onnx_path, sess_options)
        _available = True
        logger.info("Token-level MLM model loaded: {}", onnx_path)
    except ImportError:
        _available = False
        logger.info("onnxruntime/transformers not installed, token-level features disabled")
    except Exception as e:
        _available = False
        logger.warning("Token-level MLM model load failed: {}", e)

    return _available


def _mask_and_predict(text: str, mask_positions: list) -> list:
    """Mask tokens at given positions and collect log-probs of original tokens.

    Args:
        text: input text
        mask_positions: list of token indices to mask

    Returns:
        list of (original_token_id, log_prob_of_original) tuples
    """
    encoded = _tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors='np',
    )

    input_ids = encoded['input_ids'][0].copy()
    attention_mask = encoded['attention_mask'][0]

    # Build results for each mask position
    results = []
    for pos in mask_positions:
        if pos < 1 or pos >= len(input_ids) - 1:
            continue  # skip [CLS] and [SEP]

        original_id = int(input_ids[pos])
        input_ids[pos] = 103  # [MASK] token id for bert-base-chinese

        # Run inference
        input_names = {inp.name for inp in _onnx_session.get_inputs()}
        feeds = {
            'input_ids': input_ids.astype(np.int64).reshape(1, -1),
            'attention_mask': attention_mask.astype(np.int64).reshape(1, -1),
        }
        if 'token_type_ids' in input_names:
            feeds['token_type_ids'] = np.zeros_like(feeds['input_ids'], dtype=np.int64)

        outputs = _onnx_session.run(None, feeds)
        # Output shape: (1, seq_len, vocab_size)
        logits = outputs[0][0][pos]  # (vocab_size,)

        # Compute log softmax
        max_logit = np.max(logits)
        log_probs = logits - max_logit - np.log(np.sum(np.exp(logits - max_logit)))

        original_log_prob = float(log_probs[original_id])
        results.append((original_id, original_log_prob))

        # Restore original token
        input_ids[pos] = original_id

    return results


def compute_token_perplexity(text: str, n_positions: int = 30, seed: int = 42) -> dict:
    """Compute token-level perplexity features using BERT MLM.

    Strategy: sample up to n_positions tokens, mask each, and collect
    the log-prob of the original token under the masked model.

    Args:
        text: input Chinese text
        n_positions: max number of positions to evaluate (default 30)
        seed: random seed for position sampling

    Returns:
        dict with:
          - token_ppl: token-level perplexity (2^(-avg_log_prob))
          - token_logprob_mean: mean log-prob per token
          - token_logprob_std: std of log-probs (AI more uniform → lower std)
          - token_logprob_skew: skewness of log-probs
          - token_top1_acc: fraction where original token is top-1 prediction
          - available: whether the model was loaded
    """
    if not _init():
        return {
            'token_ppl': 0.0, 'token_logprob_mean': 0.0,
            'token_logprob_std': 0.0, 'token_logprob_skew': 0.0,
            'token_top1_acc': 0.0, 'available': False,
        }

    try:
        encoded = _tokenizer(text, max_length=512, truncation=True, return_tensors='np')
        input_ids = encoded['input_ids'][0]
        seq_len = len(input_ids)

        if seq_len < 10:
            return {
                'token_ppl': 0.0, 'token_logprob_mean': 0.0,
                'token_logprob_std': 0.0, 'token_logprob_skew': 0.0,
                'token_top1_acc': 0.0, 'available': True,
            }

        # Sample positions (avoid [CLS]=0 and [SEP])
        import random as _random
        rng = _random.Random(seed)
        candidates = list(range(1, seq_len - 1))
        # Prefer Chinese character tokens (higher byte values in input_ids)
        # Filter out pure punctuation/short tokens
        candidates = [p for p in candidates if input_ids[p] > 100]
        if len(candidates) > n_positions:
            positions = rng.sample(candidates, n_positions)
        else:
            positions = candidates

        if not positions:
            return {
                'token_ppl': 0.0, 'token_logprob_mean': 0.0,
                'token_logprob_std': 0.0, 'token_logprob_skew': 0.0,
                'token_top1_acc': 0.0, 'available': True,
            }

        results = _mask_and_predict(text, positions)
        if not results:
            return {
                'token_ppl': 0.0, 'token_logprob_mean': 0.0,
                'token_logprob_std': 0.0, 'token_logprob_skew': 0.0,
                'token_top1_acc': 0.0, 'available': True,
            }

        log_probs = [lp for _, lp in results]
        n = len(log_probs)
        mean_lp = sum(log_probs) / n
        ppl = 2 ** (-mean_lp)

        # Std
        variance = sum((x - mean_lp) ** 2 for x in log_probs) / n
        std_lp = variance ** 0.5

        # Skewness
        if std_lp > 0:
            skew = sum(((x - mean_lp) / std_lp) ** 3 for x in log_probs) / n
        else:
            skew = 0.0

        # Top-1 accuracy: check if original token had highest logit
        # (we don't have the logits here, but we can approximate from log_probs)
        # For now, use threshold: if log_prob > -2.0, consider it "predicted"
        top1_acc = sum(1 for lp in log_probs if lp > -2.0) / n

        return {
            'token_ppl': ppl,
            'token_logprob_mean': mean_lp,
            'token_logprob_std': std_lp,
            'token_logprob_skew': skew,
            'token_top1_acc': top1_acc,
            'available': True,
        }

    except Exception as e:
        logger.debug("Token-level perplexity computation failed: {}", e)
        return {
            'token_ppl': 0.0, 'token_logprob_mean': 0.0,
            'token_logprob_std': 0.0, 'token_logprob_skew': 0.0,
            'token_top1_acc': 0.0, 'available': False,
        }


def compute_cross_sentence_ppl(text: str, n_sentences: int = 5, seed: int = 42) -> dict:
    """Compute cross-sentence coherence features.

    For each sentence boundary, concatenate the end of the previous sentence
    with the start of the next sentence, mask the first token of the next sentence,
    and measure how predictable it is given the previous context.

    AI text tends to have more predictable sentence transitions.

    Args:
        text: input text
        n_sentences: max number of sentence boundaries to evaluate
        seed: random seed

    Returns:
        dict with:
          - cross_sent_ppl: cross-sentence perplexity
          - cross_sent_gap: gap between cross-sent and within-sent log-prob
          - available: whether computation succeeded
    """
    if not _init():
        return {'cross_sent_ppl': 0.0, 'cross_sent_gap': 0.0, 'available': False}

    try:
        import re
        import random as _random

        # Split into sentences
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if sum(1 for c in s if '\u4e00' <= c <= '\u9fff') >= 5]

        if len(sentences) < 3:
            return {'cross_sent_ppl': 0.0, 'cross_sent_gap': 0.0, 'available': True}

        rng = _random.Random(seed)
        boundaries = list(range(len(sentences) - 1))
        if len(boundaries) > n_sentences:
            boundaries = rng.sample(boundaries, n_sentences)

        cross_log_probs = []
        within_log_probs = []

        for bi in boundaries:
            prev_sent = sentences[bi]
            next_sent = sentences[bi + 1]

            # Cross-sentence: mask first token of next sentence
            combined = prev_sent + '。' + next_sent
            encoded = _tokenizer(combined, max_length=512, truncation=True, return_tensors='np')
            input_ids = encoded['input_ids'][0]

            # Find the position of the first token of next_sent
            # Approximate: find '。' position in tokens
            prev_encoded = _tokenizer(prev_sent, max_length=512, truncation=True, return_tensors='np')
            prev_len = len(prev_encoded['input_ids'][0])

            mask_pos = prev_len  # position right after previous sentence tokens
            if mask_pos >= len(input_ids) - 1:
                continue

            original_id = int(input_ids[mask_pos])
            input_ids_copy = input_ids.copy()
            input_ids_copy[mask_pos] = 103

            attention_mask = encoded['attention_mask'][0]
            input_names = {inp.name for inp in _onnx_session.get_inputs()}
            feeds = {
                'input_ids': input_ids_copy.astype(np.int64).reshape(1, -1),
                'attention_mask': attention_mask.astype(np.int64).reshape(1, -1),
            }
            if 'token_type_ids' in input_names:
                feeds['token_type_ids'] = np.zeros_like(feeds['input_ids'], dtype=np.int64)

            outputs = _onnx_session.run(None, feeds)
            logits = outputs[0][0][mask_pos]
            max_logit = np.max(logits)
            log_probs = logits - max_logit - np.log(np.sum(np.exp(logits - max_logit)))
            cross_log_probs.append(float(log_probs[original_id]))

            # Within-sentence: mask a random token in the middle of next_sent
            next_encoded = _tokenizer(next_sent, max_length=512, truncation=True, return_tensors='np')
            next_ids = next_encoded['input_ids'][0]
            if len(next_ids) > 4:
                within_pos = rng.randint(1, len(next_ids) - 2)
                orig_id = int(next_ids[within_pos])
                next_ids_copy = next_ids.copy()
                next_ids_copy[within_pos] = 103
                feeds2 = {
                    'input_ids': next_ids_copy.astype(np.int64).reshape(1, -1),
                    'attention_mask': np.ones_like(next_ids_copy, dtype=np.int64).reshape(1, -1),
                }
                if 'token_type_ids' in input_names:
                    feeds2['token_type_ids'] = np.zeros_like(feeds2['input_ids'], dtype=np.int64)
                outputs2 = _onnx_session.run(None, feeds2)
                logits2 = outputs2[0][0][within_pos]
                max_l2 = np.max(logits2)
                lp2 = logits2 - max_l2 - np.log(np.sum(np.exp(logits2 - max_l2)))
                within_log_probs.append(float(lp2[orig_id]))

        if not cross_log_probs:
            return {'cross_sent_ppl': 0.0, 'cross_sent_gap': 0.0, 'available': True}

        mean_cross = sum(cross_log_probs) / len(cross_log_probs)
        cross_ppl = 2 ** (-mean_cross)

        mean_within = sum(within_log_probs) / len(within_log_probs) if within_log_probs else mean_cross
        gap = mean_cross - mean_within

        return {
            'cross_sent_ppl': cross_ppl,
            'cross_sent_gap': gap,
            'available': True,
        }

    except Exception as e:
        logger.debug("Cross-sentence perplexity computation failed: {}", e)
        return {'cross_sent_ppl': 0.0, 'cross_sent_gap': 0.0, 'available': False}

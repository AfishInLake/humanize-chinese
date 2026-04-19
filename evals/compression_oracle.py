#!/usr/bin/env python3
"""Compression-ratio AIGC oracle (AIDetx / ZipPy approach, arxiv 2411.19869).

Idea: seed gzip with a corpus of AI text and another with human text, measure
how much extra each corpus has to "learn" to compress the candidate. AI-like
candidates compress better against an AI-seeded dictionary because they
reuse stock phrases; human-like candidates need more new dictionary entries.

Ratio = ai_marginal / human_marginal. Lower → more AI-like.

This runs zero-LLM, zero-dep (stdlib gzip only). Intended as CI oracle to
replace the captcha-blocked PaperPass / 朱雀 and avoid 850MB
AIGC_detector_zhv2 dependency.

Usage:
    python3 compression_oracle.py --calibrate   # build seeds + calibrate threshold
    python3 compression_oracle.py --score "<text>"
    python3 compression_oracle.py --benchmark   # HC3 100-sample eval
"""
import argparse
import gzip
import json
import os
import random
import sys
from statistics import mean, median, stdev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = '/Users/mac/claudeclaw/humanize'
HC3_DATA = f'{WORKSPACE}/data/hc3_chinese_all.jsonl'

AI_SEED_PATH = f'{REPO}/evals/compression_ai_seed.txt'
HUMAN_SEED_PATH = f'{REPO}/evals/compression_human_seed.txt'

# Seed size: large enough to capture AI/human stock phrases, small enough
# to keep per-call compression time sub-millisecond.
SEED_CHARS = 50000


def _gzip_len(data):
    """Compressed length in bytes. Use mtime=0 for determinism."""
    return len(gzip.compress(data.encode('utf-8'), compresslevel=6, mtime=0))


def build_seeds(n_samples=400, seed=42):
    """Build gzip seeds from HC3: SEED_CHARS of AI and SEED_CHARS of human.
    Deterministic with seed."""
    if not os.path.exists(HC3_DATA):
        raise FileNotFoundError(f'HC3 data not at {HC3_DATA}')
    rng = random.Random(seed)
    ai_texts = []
    human_texts = []
    with open(HC3_DATA, encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            for a in row.get('chatgpt_answers', []) or []:
                if a and sum(1 for c in a if '\u4e00' <= c <= '\u9fff') >= 80:
                    ai_texts.append(a)
            for h in row.get('human_answers', []) or []:
                if h and sum(1 for c in h if '\u4e00' <= c <= '\u9fff') >= 80:
                    human_texts.append(h)
    rng.shuffle(ai_texts)
    rng.shuffle(human_texts)

    ai_seed = ''
    for t in ai_texts:
        if len(ai_seed) + len(t) + 2 > SEED_CHARS:
            break
        ai_seed += t + '\n\n'
    human_seed = ''
    for t in human_texts:
        if len(human_seed) + len(t) + 2 > SEED_CHARS:
            break
        human_seed += t + '\n\n'

    with open(AI_SEED_PATH, 'w', encoding='utf-8') as f:
        f.write(ai_seed)
    with open(HUMAN_SEED_PATH, 'w', encoding='utf-8') as f:
        f.write(human_seed)
    return ai_seed, human_seed


def load_seeds():
    if not os.path.exists(AI_SEED_PATH) or not os.path.exists(HUMAN_SEED_PATH):
        return build_seeds()
    return (
        open(AI_SEED_PATH, encoding='utf-8').read(),
        open(HUMAN_SEED_PATH, encoding='utf-8').read(),
    )


_seed_ai_len = None
_seed_human_len = None
_seed_ai = None
_seed_human = None


def _ensure_seeds():
    global _seed_ai, _seed_human, _seed_ai_len, _seed_human_len
    if _seed_ai is None:
        _seed_ai, _seed_human = load_seeds()
        _seed_ai_len = _gzip_len(_seed_ai)
        _seed_human_len = _gzip_len(_seed_human)


def score(text):
    """Return compression ratio: ai_marginal / human_marginal.
    < 1.0 = AI-like (compresses better vs AI seed)
    > 1.0 = human-like
    Returns float, or None if text too short (< 50 chars).
    """
    if not text or len(text) < 50:
        return None
    _ensure_seeds()
    ai_combined = _gzip_len(_seed_ai + text)
    human_combined = _gzip_len(_seed_human + text)
    ai_marginal = max(1, ai_combined - _seed_ai_len)
    human_marginal = max(1, human_combined - _seed_human_len)
    return ai_marginal / human_marginal


def calibrate(n=300, seed=42):
    """Compute Cohen's d between AI and human on HC3 holdout."""
    _ensure_seeds()
    rng = random.Random(seed)
    ai_scores = []
    human_scores = []
    with open(HC3_DATA, encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            for a in row.get('chatgpt_answers', []) or []:
                if a and sum(1 for c in a if '\u4e00' <= c <= '\u9fff') >= 100:
                    ai_scores.append(a)
            for h in row.get('human_answers', []) or []:
                if h and sum(1 for c in h if '\u4e00' <= c <= '\u9fff') >= 100:
                    human_scores.append(h)
    rng.shuffle(ai_scores)
    rng.shuffle(human_scores)
    ai_sub = ai_scores[:n]
    human_sub = human_scores[:n]

    ai_r = [score(t) for t in ai_sub]
    ai_r = [r for r in ai_r if r is not None]
    human_r = [score(t) for t in human_sub]
    human_r = [r for r in human_r if r is not None]
    return ai_r, human_r


def _cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    sa, sb = stdev(a), stdev(b)
    pooled = ((sa**2 + sb**2) / 2) ** 0.5
    return (mb - ma) / pooled if pooled > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-seeds', action='store_true')
    ap.add_argument('--score', type=str, help='score this text')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()

    if args.build_seeds:
        ai_seed, human_seed = build_seeds()
        print(f'AI seed: {len(ai_seed)} chars -> {AI_SEED_PATH}')
        print(f'Human seed: {len(human_seed)} chars -> {HUMAN_SEED_PATH}')
        return

    if args.score:
        r = score(args.score)
        print(f'compression_ratio = {r:.4f} (< 1.0 = AI-like)')
        return

    if args.calibrate:
        print(f'Calibrating with n={args.n} HC3 AI + {args.n} human samples...')
        ai_r, human_r = calibrate(n=args.n)
        d = _cohen_d(ai_r, human_r)
        print(f'AI ratio:    mean={mean(ai_r):.4f}  median={median(ai_r):.4f}  n={len(ai_r)}')
        print(f'Human ratio: mean={mean(human_r):.4f}  median={median(human_r):.4f}  n={len(human_r)}')
        print(f"Cohen's d = {d:.3f}")
        # Decision threshold: midpoint between means
        thr = (mean(ai_r) + mean(human_r)) / 2
        ai_flagged = sum(1 for r in ai_r if r < thr)
        human_not = sum(1 for r in human_r if r >= thr)
        print(f'Threshold {thr:.4f}: flags {ai_flagged}/{len(ai_r)} AI ({100*ai_flagged/len(ai_r):.0f}%), '
              f'correctly passes {human_not}/{len(human_r)} human ({100*human_not/len(human_r):.0f}%)')
        return

    ap.print_help()


if __name__ == '__main__':
    main()

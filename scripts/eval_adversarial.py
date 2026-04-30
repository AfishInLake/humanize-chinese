#!/usr/bin/env python3
"""
抗攻击测试: 模拟常见的 AI 文本伪装手段，评估检测器鲁棒性。

攻击类型:
1. 同义词替换 (10-20% 词)
2. 句子打乱
3. 插入噪声 (语气词/停顿)
4. 混合文本 (人类+AI 段落交替)
5. LLM 改写 (用另一个模型改写 AI 文本)

Usage:
    python scripts/eval_adversarial.py --test scripts/training_data/test.jsonl
    python scripts/eval_adversarial.py --test scripts/training_data/test.jsonl --attack synonym
    python scripts/eval_adversarial.py --test scripts/training_data/test.jsonl --attack all
"""

import argparse
import json
import os
import random
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

DEFAULT_TEST = os.path.join(SCRIPT_DIR, 'training_data', 'test.jsonl')


def load_jsonl(filepath, label=1, max_samples=None):
    """Load texts with specific label from JSONL."""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if int(row.get('label', -1)) == label:
                    texts.append(row.get('text', ''))
                    if max_samples and len(texts) >= max_samples:
                        break
            except json.JSONDecodeError:
                continue
    return texts


# ─── Attack functions ───

def attack_synonym_replace(text, replace_ratio=0.15, seed=42):
    """同义词替换攻击: 用 CiLin 同义词替换 15% 的词。"""
    rng = random.Random(seed)

    # Load CiLin
    cilin_path = os.path.join(REPO_ROOT, 'src', 'humanize_cn', 'data',
                               'cilin_synonyms.json')
    if not os.path.exists(cilin_path):
        return text

    with open(cilin_path, 'r', encoding='utf-8') as f:
        cilin = json.load(f)

    # Tokenize
    try:
        import jieba
        words = list(jieba.cut(text))
    except ImportError:
        return text

    # Find replaceable words
    replaceable = []
    for w in words:
        if len(w) >= 2 and sum(1 for c in w if '\u4e00' <= c <= '\u9fff') >= 2:
            syns = cilin.get(w, [])
            if isinstance(syns, str):
                syns = [syns]
            syns = [s for s in syns if s != w and len(s) > 0]
            if syns:
                replaceable.append((w, rng.choice(syns)))

    if not replaceable:
        return text

    n_replace = max(1, int(len(replaceable) * replace_ratio))
    to_replace = dict(rng.sample(replaceable, min(n_replace, len(replaceable))))

    result = text
    for orig, repl in to_replace.items():
        result = result.replace(orig, repl, 1)

    return result


def attack_sentence_shuffle(text, seed=42):
    """句子打乱攻击: 随机打乱句子顺序。"""
    rng = random.Random(seed)
    sentences = re.split(r'([。！？\n])', text)
    # Group sentence + punctuation
    pairs = []
    for i in range(0, len(sentences) - 1, 2):
        pairs.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:
        pairs.append(sentences[-1])

    if len(pairs) <= 2:
        return text

    rng.shuffle(pairs)
    return ''.join(pairs)


def attack_insert_noise(text, noise_ratio=0.05, seed=42):
    """插入噪声攻击: 在句末插入语气词/停顿词。"""
    rng = random.Random(seed)
    noise_words = ['嗯', '啊', '呢', '吧', '哦', '嘛', '呀', '哈',
                   '其实吧', '说实话', '怎么说呢', '老实说', '坦白讲',
                   '你知道吗', '我觉得吧', '说真的']

    sentences = re.split(r'([。！？])', text)
    result = []
    for i, sent in enumerate(sentences):
        result.append(sent)
        # Insert noise after punctuation (every few sentences)
        if sent in '。！？' and rng.random() < noise_ratio * 10:
            if rng.random() < 0.5:
                result.append(rng.choice(noise_words))
            else:
                result.append('，')

    return ''.join(result)


def attack_mixed_text(ai_text, human_texts, mix_ratio=0.3, seed=42):
    """混合文本攻击: 人类文本中插入 AI 段落。"""
    if not human_texts:
        return ai_text

    rng = random.Random(seed)
    ai_sentences = re.split(r'([。！？\n])', ai_text)
    # Group into sentence pairs
    ai_groups = []
    for i in range(0, len(ai_sentences) - 1, 2):
        ai_groups.append(ai_sentences[i] + ai_sentences[i+1])

    # Pick random human text and split
    human_text = rng.choice(human_texts)
    human_sentences = re.split(r'([。！？\n])', human_text)
    human_groups = []
    for i in range(0, len(human_sentences) - 1, 2):
        human_groups.append(human_sentences[i] + human_sentences[i+1])

    if not human_groups:
        return ai_text

    # Mix: replace some AI sentences with human sentences
    n_replace = max(1, int(len(ai_groups) * mix_ratio))
    replace_indices = rng.sample(range(len(ai_groups)),
                                  min(n_replace, len(ai_groups)))
    for idx in replace_indices:
        ai_groups[idx] = rng.choice(human_groups)

    return ''.join(ai_groups)


def apply_attack(text, attack_type, seed=42, human_texts=None):
    """Apply specified attack to text."""
    if attack_type == 'synonym':
        return attack_synonym_replace(text, seed=seed)
    elif attack_type == 'shuffle':
        return attack_sentence_shuffle(text, seed=seed)
    elif attack_type == 'noise':
        return attack_insert_noise(text, seed=seed)
    elif attack_type == 'mixed':
        return attack_mixed_text(text, human_texts or [], seed=seed)
    else:
        return text


def main():
    parser = argparse.ArgumentParser(description='抗攻击测试脚本')
    parser.add_argument('--test', type=str, default=DEFAULT_TEST,
                        help='测试数据路径 (JSONL)')
    parser.add_argument('--attack', type=str, default='all',
                        choices=['all', 'synonym', 'shuffle', 'noise', 'mixed'],
                        help='攻击类型 (默认: all)')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='最大测试样本数 (默认: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    print(f'{"="*60}')
    print('抗攻击测试脚本')
    print(f'{"="*60}')
    print(f'  测试数据: {args.test}')
    print(f'  攻击类型: {args.attack}')
    print(f'  最大样本: {args.max_samples}')
    print(f'{"="*60}\n')

    # Load AI texts and human texts (for mixed attack)
    ai_texts = load_jsonl(args.test, label=1, max_samples=args.max_samples)
    human_texts = load_jsonl(args.test, label=0, max_samples=args.max_samples)

    if not ai_texts:
        print('错误: 未找到 AI 文本样本 (label=1)')
        sys.exit(1)

    print(f'AI 样本: {len(ai_texts)} 条')
    print(f'Human 样本: {len(human_texts)} 条')

    from humanize_cn.check_pkg.api import check

    attacks = {
        'synonym': '同义词替换',
        'shuffle': '句子打乱',
        'noise': '插入噪声',
        'mixed': '混合文本',
    }

    if args.attack == 'all':
        attack_list = list(attacks.keys())
    else:
        attack_list = [args.attack]

    # Baseline: score original AI texts
    print(f'\n{"="*60}')
    print('基线: 原始 AI 文本检测')
    print(f'{"="*60}')

    baseline_scores = []
    for text in ai_texts:
        r = check(text)
        baseline_scores.append(r['ai_score'])

    baseline_detected = sum(1 for s in baseline_scores if s >= 50)
    baseline_rate = baseline_detected / len(baseline_scores) * 100
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    print(f'  检测率: {baseline_rate:.1f}% (≥50分)')
    print(f'  平均分: {baseline_mean:.1f}')

    # Run attacks
    results = {}
    for attack_name in attack_list:
        print(f'\n{"="*60}')
        print(f'攻击: {attacks[attack_name]}')
        print(f'{"="*60}')

        attack_scores = []
        for i, text in enumerate(ai_texts):
            attacked = apply_attack(text, attack_name, seed=args.seed + i,
                                    human_texts=human_texts)
            r = check(attacked)
            attack_scores.append(r['ai_score'])

            if (i + 1) % 20 == 0:
                print(f'  进度: {i+1}/{len(ai_texts)}')

        detected = sum(1 for s in attack_scores if s >= 50)
        rate = detected / len(attack_scores) * 100
        mean_score = sum(attack_scores) / len(attack_scores)
        drop = baseline_rate - rate

        print(f'  检测率: {rate:.1f}% (≥50分)')
        print(f'  平均分: {mean_score:.1f}')
        print(f'  下降:   {drop:.1f} 百分点')

        results[attack_name] = {
            'detection_rate': rate,
            'mean_score': mean_score,
            'drop': drop,
        }

    # Summary
    print(f'\n{"="*60}')
    print('抗攻击测试总结')
    print(f'{"="*60}')
    print(f'  基线检测率: {baseline_rate:.1f}%')
    for name, res in results.items():
        status = '✓ 通过' if res['drop'] < 15 else '✗ 需改进'
        print(f'  {attacks[name]:10s}: {res["detection_rate"]:.1f}% '
              f'(下降 {res["drop"]:.1f}%) {status}')

    # Save results
    out_path = os.path.join(os.path.dirname(args.test), 'adversarial_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {
                'detection_rate': baseline_rate,
                'mean_score': baseline_mean,
            },
            'attacks': results,
            'n_samples': len(ai_texts),
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_path}')


if __name__ == '__main__':
    main()

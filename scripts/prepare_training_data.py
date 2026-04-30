#!/usr/bin/env python3
"""
数据清洗与划分脚本。

合并所有数据源（HC3 人类文本 + HC3 ChatGPT + API 生成 + 公开数据集），
清洗、去重、平衡、划分 train/val/test。

用法:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --min-cn 100 --max-cn 5000
    python scripts/prepare_training_data.py --ai-generated ./training_data/ai_generated_v1.jsonl

输出:
    scripts/training_data/train.jsonl
    scripts/training_data/val.jsonl
    scripts/training_data/test.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_HC3 = os.path.join(SCRIPT_DIR, 'training_data', 'hc3_all.jsonl')
DEFAULT_AI_GEN_DIR = os.path.join(SCRIPT_DIR, 'training_data')
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')


def count_cn_chars(text: str) -> int:
    """统计中文字符数。"""
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')


def clean_text(text: str) -> str:
    """清洗文本。"""
    if not text:
        return ''
    # 去除零宽字符
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    # 去除 \r\n 字面量字符串（转义残留）
    text = text.replace('\\r\\n', ' ').replace('\\r', ' ').replace('\\n', ' ')
    # 去除实际换行符
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    # 合并连续空白
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def text_hash(text: str) -> str:
    """计算文本的 simhash（简化版：取前 200 字符的 MD5）。"""
    normalized = re.sub(r'\s+', '', text)[:200]
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def load_hc3_data(filepath: str, min_cn: int = 100) -> list:
    """从 HC3-Chinese JSONL 加载人类和 AI 文本。"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # 人类回答
                for ans in row.get('human_answers', []) or []:
                    ans = clean_text(ans)
                    if count_cn_chars(ans) >= min_cn:
                        samples.append({
                            'text': ans,
                            'label': 0,
                            'source': 'hc3_human',
                            'style': 'qa',
                        })
                # ChatGPT 回答
                for ans in row.get('chatgpt_answers', []) or []:
                    ans = clean_text(ans)
                    if count_cn_chars(ans) >= min_cn:
                        samples.append({
                            'text': ans,
                            'label': 1,
                            'source': 'hc3_chatgpt',
                            'style': 'qa',
                        })
            except json.JSONDecodeError:
                continue
    return samples


def load_ai_generated(data_dir: str, min_cn: int = 100) -> list:
    """从 training_data/ 下所有 ai_generated*.jsonl 加载 AI 文本。"""
    if not os.path.isdir(data_dir):
        print(f'[跳过] 目录不存在: {data_dir}')
        return []

    samples = []
    jsonl_files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith('ai_generated') and f.endswith('.jsonl')
    )
    if not jsonl_files:
        print(f'[跳过] {data_dir} 下没有 ai_generated*.jsonl 文件')
        return []

    print(f'  扫描到 {len(jsonl_files)} 个 AI 生成文件:')
    for fname in jsonl_files:
        fpath = os.path.join(data_dir, fname)
        count_before = len(samples)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = clean_text(row.get('ai_answer', ''))
                    if count_cn_chars(text) >= min_cn:
                        samples.append({
                            'text': text,
                            'label': 1,
                            'source': f"api_{row.get('model', 'unknown')}",
                            'style': row.get('style', 'direct'),
                        })
                except json.JSONDecodeError:
                    continue
        print(f'    {fname}: +{len(samples) - count_before} 条')
    return samples


def deduplicate(samples: list) -> list:
    """去重：基于文本 hash。"""
    seen = set()
    unique = []
    dup_count = 0
    for s in samples:
        h = text_hash(s['text'])
        if h not in seen:
            seen.add(h)
            unique.append(s)
        else:
            dup_count += 1
    print(f'[去重] 移除 {dup_count} 条重复文本，剩余 {len(unique)} 条')
    return unique


def balance_samples(samples: list, seed: int = 42) -> list:
    """平衡 AI/human 样本数量（欠采样多数类）。"""
    rng = random.Random(seed)
    ai = [s for s in samples if s['label'] == 1]
    human = [s for s in samples if s['label'] == 0]

    print(f'[平衡] AI: {len(ai)}, Human: {len(human)}')

    target = min(len(ai), len(human))
    if len(ai) > target:
        rng.shuffle(ai)
        ai = ai[:target]
        print(f'[平衡] AI 欠采样到 {target} 条')
    elif len(human) > target:
        rng.shuffle(human)
        human = human[:target]
        print(f'[平衡] Human 欠采样到 {target} 条')

    return ai + human


def split_data(samples: list, ratios=(0.7, 0.15, 0.15), seed: int = 42):
    """划分 train/val/test，保持类别比例。"""
    rng = random.Random(seed)

    # 按类别分组
    ai = [s for s in samples if s['label'] == 1]
    human = [s for s in samples if s['label'] == 0]

    def split_list(lst, ratios):
        rng.shuffle(lst)
        n = len(lst)
        splits = []
        start = 0
        for r in ratios[:-1]:
            end = start + int(n * r)
            splits.append(lst[start:end])
            start = end
        splits.append(lst[start:])
        return splits

    ai_splits = split_list(ai, ratios)
    human_splits = split_list(human, ratios)

    train = ai_splits[0] + human_splits[0]
    val = ai_splits[1] + human_splits[1]
    test = ai_splits[2] + human_splits[2]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def print_stats(samples: list, name: str):
    """打印数据集统计信息。"""
    n = len(samples)
    n_ai = sum(1 for s in samples if s['label'] == 1)
    n_human = n - n_ai
    sources = Counter(s['source'] for s in samples)
    styles = Counter(s['style'] for s in samples)
    lengths = [count_cn_chars(s['text']) for s in samples]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f'\n[{name}] {n} 条 (AI: {n_ai}, Human: {n_human})')
    print(f'  平均长度: {avg_len:.0f} 字符')
    print(f'  来源分布: {dict(sources.most_common(10))}')
    print(f'  风格分布: {dict(styles.most_common())}')


def main():
    parser = argparse.ArgumentParser(description='数据清洗与划分脚本')
    parser.add_argument('--hc3', type=str, default=DEFAULT_HC3,
                        help='HC3 数据路径')
    parser.add_argument('--ai-generated-dir', type=str, default=DEFAULT_AI_GEN_DIR,
                        help='AI 生成文件所在目录，自动扫描 ai_generated*.jsonl')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--min-cn', type=int, default=100,
                        help='最小中文字符数 (默认: 100)')
    parser.add_argument('--max-cn', type=int, default=5000,
                        help='最大中文字符数 (默认: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-balance', action='store_true',
                        help='不进行类别平衡')
    args = parser.parse_args()

    print(f'{"="*60}')
    print('数据清洗与划分脚本')
    print(f'{"="*60}')
    print(f'  HC3:         {args.hc3}')
    print(f'  AI 生成目录: {args.ai_generated_dir}')
    print(f'  输出目录:    {args.output_dir}')
    print(f'  字符范围:    {args.min_cn}-{args.max_cn}')
    print(f'{"="*60}\n')

    # 1. 加载数据
    print('[步骤 1/4] 加载数据...')
    samples = []

    hc3_samples = load_hc3_data(args.hc3, args.min_cn)
    samples.extend(hc3_samples)
    print(f'  HC3: {len(hc3_samples)} 条')

    ai_gen_samples = load_ai_generated(args.ai_generated_dir, args.min_cn)
    samples.extend(ai_gen_samples)
    print(f'  API 生成: {len(ai_gen_samples)} 条')

    print(f'  合计: {len(samples)} 条')

    # 2. 过滤长度
    print('\n[步骤 2/4] 过滤长度...')
    before = len(samples)
    samples = [s for s in samples
               if args.min_cn <= count_cn_chars(s['text']) <= args.max_cn]
    print(f'  过滤 {before - len(samples)} 条，剩余 {len(samples)} 条')

    # 3. 去重
    print('\n[步骤 3/4] 去重...')
    samples = deduplicate(samples)

    # 4. 平衡
    if not args.no_balance:
        print('\n[步骤 3.5/4] 平衡...')
        samples = balance_samples(samples, args.seed)

    # 5. 划分
    print('\n[步骤 4/4] 划分 train/val/test...')
    train, val, test = split_data(samples, seed=args.seed)

    # 6. 保存
    os.makedirs(args.output_dir, exist_ok=True)

    for name, data in [('train', train), ('val', val), ('test', test)]:
        outpath = os.path.join(args.output_dir, f'{name}.jsonl')
        with open(outpath, 'w', encoding='utf-8') as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print_stats(data, name)

    # 7. 总结
    print(f'\n{"="*60}')
    print('数据准备完成!')
    print(f'{"="*60}')
    print(f'  训练集: {len(train)} 条')
    print(f'  验证集: {len(val)} 条')
    print(f'  测试集: {len(test)} 条')
    print(f'  总计:   {len(train) + len(val) + len(test)} 条')
    print(f'  输出目录: {args.output_dir}')


if __name__ == '__main__':
    main()

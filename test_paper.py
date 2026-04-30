#!/usr/bin/env python3
"""论文 AIGC 检测脚本（三路融合版）

读取 JSON 文件中的段落，逐段使用 BERT + XGBoost + Rule 三路融合检测，
输出 AI 评分和详细指标。

运行方式:
    PYTHONPATH=src python3 test_paper.py 论文.json
    PYTHONPATH=src python3 test_paper.py 论文.json --no-rewrite   # 不做改写
"""

import argparse
import json
import sys
import os

# 在 import transformers 之前设置，抑制警告
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
import warnings
warnings.filterwarnings('ignore', message='.*PyTorch.*TensorFlow.*Flax.*')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from humanize_cn.check_pkg.api import check

# 跳过的类别（标题、关键词、图表标题、参考文献等）
SKIP_CATEGORIES = {
    'abstract_chinese_title', 'keywords_chinese', 'keywords_english',
    'heading_level_1', 'heading_level_2', 'heading_level_3',
    'caption_figure', 'caption_table',
    'references_title', 'acknowledgements_title',
}

# 最短检测长度（中文字符 < 20 的跳过）
MIN_CHINESE_CHARS = 20


def count_chinese(text):
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')


def load_paper(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='论文 AIGC 检测（三路融合）')
    parser.add_argument('paper', nargs='?', default='校园二手物品交易平台的设计与实现.json',
                        help='论文 JSON 文件路径')
    parser.add_argument('--no-rewrite', action='store_true',
                        help='跳过改写对比')
    args = parser.parse_args()

    data = load_paper(args.paper)

    print(f"{'='*70}")
    print(f"  论文 AIGC 检测报告（BERT + XGBoost + Rule 三路融合）")
    print(f"  文件: {args.paper}")
    print(f"  段落总数: {len(data)}")
    print(f"{'='*70}")

    results = []

    for i, item in enumerate(data):
        cat = item.get('category', '')
        para = item.get('paragraph', '').strip()

        # 跳过非正文
        if cat in SKIP_CATEGORIES:
            continue
        if count_chinese(para) < MIN_CHINESE_CHARS:
            continue

        # 三路融合检测
        r = check(para)

        results.append({
            'index': i,
            'category': cat,
            'paragraph': para,
            'ai_score': r['ai_score'],
            'ai_level': r['ai_level'],
            'ai_method': r['ai_method'],
            'perplexity': r.get('perplexity'),
            'burstiness': r.get('burstiness'),
            'indicators': r.get('indicators', {}),
        })

        print(f"  段落 #{i+1}: AI={r['ai_score']:3d}/100 ({r['ai_level']}) "
              f"方法={r['ai_method']} "
              f"ppl={r.get('perplexity', 'N/A')}")

    if not results:
        print("\n没有可检测的段落。")
        return

    # 按分数排序
    results.sort(key=lambda x: x['ai_score'], reverse=True)

    # ─── 汇总统计 ───
    scores = [r['ai_score'] for r in results]
    avg = sum(scores) / len(scores)
    methods = set(r['ai_method'] for r in results)

    high_count = sum(1 for s in scores if s >= 75)
    mid_high_count = sum(1 for s in scores if 50 <= s < 75)
    mid_count = sum(1 for s in scores if 25 <= s < 50)
    low_count = sum(1 for s in scores if s < 25)

    print(f"\n{'='*70}")
    print(f"  汇总统计")
    print(f"{'='*70}")
    print(f"  检测段落数: {len(results)}")
    print(f"  平均 AI 分: {avg:.1f}")
    print(f"  使用方法:   {', '.join(methods)}")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │ VERY HIGH (≥75): {high_count:3d} 段                 │")
    print(f"  │ HIGH      (50-74): {mid_high_count:3d} 段               │")
    print(f"  │ MEDIUM    (25-49): {mid_count:3d} 段               │")
    print(f"  │ LOW       (<25): {low_count:3d} 段                │")
    print(f"  └─────────────────────────────────────────┘")

    # ─── 逐段详情（按 AI 分数降序）───
    print(f"\n{'='*70}")
    print(f"  逐段详情（按 AI 分数降序，只显示 ≥25 分）")
    print(f"{'='*70}")

    for r in results:
        if r['ai_score'] < 25:
            continue

        bar_len = 20
        filled = int(r['ai_score'] / 100 * bar_len)
        bar = '█' * filled + '░' * (bar_len - filled)

        print(f"\n── 段落 #{r['index']+1} ──")
        print(f"  [{r['ai_score']:3d}/100 {bar}] {r['ai_level']}")
        print(f"  检测方法: {r['ai_method']}")
        if r['perplexity'] is not None:
            print(f"  困惑度: {r['perplexity']:.2f}  突发性: {r.get('burstiness', 'N/A')}")
        print(f"  原文: {r['paragraph'][:100]}...")

        hit_keys = [k for k, v in r['indicators'].items() if v]
        if hit_keys:
            print(f"  命中指标: {', '.join(hit_keys[:8])}")

    # ─── 改写对比（高风险段落）───
    if not args.no_rewrite:
        high_risk = [r for r in results if r['ai_score'] >= 50]
        if high_risk:
            print(f"\n{'='*70}")
            print(f"  高风险段落改写对比（AI ≥ 50）")
            print(f"{'='*70}")

            try:
                from humanize_cn import humanize_academic
                can_rewrite = True
            except ImportError:
                print("  ⚠️  humanize_academic 不可用，跳过改写")
                can_rewrite = False

            if can_rewrite:
                for r in high_risk[:5]:  # 最多展示 5 段
                    print(f"\n{'─'*60}")
                    print(f"段落 #{r['index']+1}  [AI {r['ai_score']}/100 {r['ai_level']}]")
                    print(f"原文: {r['paragraph'][:120]}...")

                    try:
                        rewritten = humanize_academic(r['paragraph'], aggressive=False, best_of_n=0)
                        new_r = check(rewritten)
                        diff = r['ai_score'] - new_r['ai_score']
                        print(f"\n  【改写后 AI={new_r['ai_score']}/100 ({new_r['ai_level']}) 降幅 {diff:+d}】")
                        print(f"  改写: {rewritten[:120]}...")
                    except Exception as e:
                        print(f"  改写失败: {e}")

    print(f"\n{'='*70}")
    print(f"  检测完毕")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

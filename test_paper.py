#!/usr/bin/env python3
"""论文 AIGC 检测 + 改写对比脚本

读取 JSON 文件中的段落，逐段检测 AI 评分，对高分段落给出改写建议。
运行方式: PYTHONPATH=src python3 test_paper.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from humanize_cn import (
    detect, calculate_score, score_to_level,
    humanize,
    detect_academic, calculate_academic_score, humanize_academic,
)

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
    paper_path = sys.argv[1] if len(sys.argv) > 1 else '毕业设计说明书.json'
    data = load_paper(paper_path)

    print(f"{'='*70}")
    print(f"  论文 AIGC 检测报告")
    print(f"  文件: {paper_path}")
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
        # 跳过纯英文段落
        if count_chinese(para) < 5:
            continue

        # 通用检测
        issues, metrics = detect(para)
        score = calculate_score(issues, metrics)
        level = score_to_level(score)

        # 学术检测
        ac_issues, ac_metrics = detect_academic(para)
        ac_score = calculate_academic_score(ac_issues)

        results.append({
            'index': i,
            'category': cat,
            'paragraph': para,
            'score': score,
            'level': level,
            'academic_score': ac_score,
            'issues': issues,
        })

    # 按分数排序
    results.sort(key=lambda x: x['score'], reverse=True)

    # ─── 汇总统计 ───
    scores = [r['score'] for r in results]
    ac_scores = [r['academic_score'] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    ac_avg = sum(ac_scores) / len(ac_scores) if ac_scores else 0
    high_count = sum(1 for s in scores if s >= 35)
    mid_count = sum(1 for s in scores if 15 <= s < 35)
    low_count = sum(1 for s in scores if s < 15)

    print(f"\n── 汇总 ──")
    print(f"  检测段落数: {len(results)}")
    print(f"  通用检测平均分: {avg:.1f}")
    print(f"  学术检测平均分: {ac_avg:.1f}")
    print(f"  高风险 (≥35): {high_count} 段  |  中风险 (15-35): {mid_count} 段  |  低风险 (<15): {low_count} 段")

    # ─── 逐段详情 ───
    print(f"\n{'='*70}")
    print(f"  逐段检测详情（按 AI 分数降序）")
    print(f"{'='*70}")

    for r in results:
        bar_len = 20
        filled = int(r['score'] / 100 * bar_len)
        bar = '█' * filled + '░' * (bar_len - filled)

        print(f"\n── 段落 #{r['index']+1} ──")
        print(f"  [{r['score']:3d}/100 {bar}] {r['level'].upper()}")
        print(f"  学术评分: {r['academic_score']}/100")
        print(f"  原文: {r['paragraph'][:80]}...")

        hit_keys = [k for k, v in r['issues'].items() if v]
        if hit_keys:
            print(f"  命中: {', '.join(hit_keys[:5])}")

        # 高分段落给出改写
        if r['score'] >= 15:
            rewritten = humanize_academic(r['paragraph'], aggressive=False, best_of_n=0)
            new_issues, _ = detect_academic(rewritten)
            new_score = calculate_academic_score(new_issues)
            diff = r['academic_score'] - new_score

            print(f"  改写: {rewritten[:80]}...")
            print(f"  改写后学术评分: {new_score}/100 (降幅 {diff:+d})")

    # ─── 高风险段落改写对比 ───
    high_risk = [r for r in results if r['score'] >= 35]
    if high_risk:
        print(f"\n{'='*70}")
        print(f"  高风险段落改写对比（通用检测 ≥ 35）")
        print(f"{'='*70}")

        for r in high_risk:
            print(f"\n{'─'*60}")
            print(f"段落 #{r['index']+1}  [原始 {r['score']}/100]")

            # 学术改写
            rw = humanize_academic(r['paragraph'], aggressive=False, best_of_n=0)
            new_score = calculate_score(*detect(rw))
            print(f"\n  【改写后 {new_score}/100  降幅 {r['score']-new_score}】")
            print(f"  {rw}")

    print(f"\n{'='*70}")
    print(f"  检测完毕")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

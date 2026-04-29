#!/usr/bin/env python3
"""
逐段落独立改写脚本。

将文本按段落拆分，每个段落独立调用 humanize() 改写，
最后拼接回去。适用于论文等长文本的段落级降重。

用法:
    python paragraph_humanize.py input.txt -o output.txt
    python paragraph_humanize.py input.txt -o output.txt --scene formal -a
    python paragraph_humanize.py input.txt -o output.txt --seed 42 --quick

与 humanize rewrite 的区别:
  - rewrite: 把整篇文本当成一个整体改写（跨段落优化）
  - paragraph_humanize: 逐段落独立改写（段落内降重，不影响其他段落）
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from .humanize import humanize
from ..check_pkg.detect import detect_patterns, calculate_score
from ..models.ngram import compute_lr_score, analyze_text


def split_paragraphs(text: str) -> list[tuple[str, str]]:
    """将文本拆分为段落，保留分隔符。

    Returns:
        list of (paragraph_text, separator) tuples
    """
    # 按双换行拆分，保留分隔符
    parts = re.split(r'(\n\n+)', text)
    paragraphs = []
    i = 0
    while i < len(parts):
        para = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ''
        if para.strip():
            paragraphs.append((para, sep))
        i += 2
    return paragraphs


def is_skippable(paragraph: str) -> bool:
    """判断段落是否应该跳过改写。"""
    text = paragraph.strip()
    if not text:
        return True
    # 纯数字/符号
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cn_chars < 20:
        return True
    # 参考文献格式
    if re.match(r'^\s*\[\d+\]', text):
        return True
    # 纯英文段落（不处理）
    if cn_chars < len(text) * 0.15:
        return True
    # 图表标题
    if re.match(r'^(图|表|Figure|Table)\s*\d', text):
        return True
    return False


def detect_paragraph(text: str) -> dict:
    """对单个段落进行 AI 检测。"""
    issues, metrics = detect_patterns(text)
    rule_score = calculate_score(issues, metrics)
    char_count = metrics.get('char_count', 0)

    # 短段落用纯规则评分
    if char_count < 100:
        score = rule_score
    else:
        try:
            lr_result = compute_lr_score(text, scene='general')
            if lr_result:
                score = round(0.2 * rule_score + 0.8 * lr_result['score'])
            else:
                score = rule_score
        except Exception:
            score = rule_score

    return {
        'score': score,
        'rule_score': rule_score,
        'char_count': char_count,
        'issues': len(issues),
    }


def humanize_paragraph(text: str, scene: str = 'formal', aggressive: bool = False,
                       seed: int = None, quick: bool = False) -> str:
    """对单个段落进行改写。"""
    return humanize(
        text,
        scene=scene,
        aggressive=aggressive,
        seed=seed,
        best_of_n=None,  # 逐段落模式不用 best-of-n（太慢）
    )


def format_score_bar(score: int, width: int = 20) -> str:
    """生成分数条。"""
    filled = int(score / 100 * width)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}]'


def score_level(score: int) -> str:
    if score >= 75:
        return 'VERY HIGH'
    elif score >= 50:
        return 'HIGH'
    elif score >= 25:
        return 'MEDIUM'
    else:
        return 'LOW'


def main():
    parser = argparse.ArgumentParser(
        description='逐段落独立改写（段落内降重）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.txt -o output.txt
  %(prog)s input.txt -o output.txt --scene formal -a
  %(prog)s input.txt -o output.txt --seed 42 --quick
  %(prog)s input.txt -o output.txt --min-score 30
        """,
    )
    parser.add_argument('file', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认输出到 stdout）')
    parser.add_argument('--scene', default='formal',
                        choices=['general', 'social', 'tech', 'formal', 'chat'],
                        help='场景（默认 formal）')
    parser.add_argument('-a', '--aggressive', action='store_true',
                        help='激进模式')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认 42）')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（跳过统计优化）')
    parser.add_argument('--min-score', type=int, default=0,
                        help='只改写 AI 评分 >= 此值的段落（默认 0，全部改写）')
    parser.add_argument('--no-detect', action='store_true',
                        help='跳过改写前检测，直接改写所有段落')
    parser.add_argument('--json', action='store_true',
                        help='输出 JSON 格式（含每个段落的改写前后评分）')

    args = parser.parse_args()

    # 读取输入
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f'错误: 无法读取文件: {e}', file=sys.stderr)
        sys.exit(1)

    # 拆分段落
    paragraphs = split_paragraphs(text)
    print(f'输入文件: {args.file}')
    print(f'总段落数: {len(paragraphs)}')
    print(f'场景: {args.scene} | 激进: {args.aggressive} | 种子: {args.seed}')
    print(f'最低改写分数: {args.min_score}')
    print()

    # 逐段落处理
    results = []
    total_before = 0
    total_after = 0
    rewritten_count = 0
    skipped_count = 0
    start_time = time.time()

    for i, (para_text, sep) in enumerate(paragraphs):
        para_label = f'段落 {i + 1}'

        # 跳过不需要改写的段落
        if is_skippable(para_text):
            results.append((para_text, sep, None, None))
            skipped_count += 1
            continue

        # 改写前检测
        if not args.no_detect:
            before = detect_paragraph(para_text)
            before_score = before['score']
        else:
            before_score = -1

        total_before += max(0, before_score)

        # 判断是否需要改写
        if not args.no_detect and before_score < args.min_score:
            results.append((para_text, sep, before_score, before_score))
            skipped_count += 1
            print(f'  {para_label}: {before_score:3d}/100 {format_score_bar(before_score)} {score_level(before_score):9s} → 跳过（低于阈值）')
            continue

        # 改写
        try:
            rewritten = humanize_paragraph(
                para_text,
                scene=args.scene,
                aggressive=args.aggressive,
                seed=args.seed + i if args.seed else None,
            )
        except Exception as e:
            print(f'  {para_label}: 改写出错: {e}，保留原文', file=sys.stderr)
            results.append((para_text, sep, before_score, before_score))
            skipped_count += 1
            continue

        # 改写后检测
        after = detect_paragraph(rewritten)
        after_score = after['score']
        total_after += max(0, after_score)

        delta = after_score - before_score
        delta_str = f'{delta:+d}' if before_score >= 0 else '?'
        status = '✓' if after_score < before_score else '→'

        cn = sum(1 for c in para_text if '\u4e00' <= c <= '\u9fff')
        print(f'  {para_label}: {before_score:3d} → {after_score:3d} ({delta_str}) {format_score_bar(after_score)} {score_level(after_score):9s} | {cn}字')

        results.append((rewritten, sep, before_score, after_score))
        rewritten_count += 1

    elapsed = time.time() - start_time

    # 拼接输出
    output_text = ''
    for para_text, sep, before_score, after_score in results:
        output_text += para_text + sep

    # 输出结果
    print()
    print(f'─' * 50)
    print(f'处理完成: {rewritten_count} 个段落改写, {skipped_count} 个段落跳过')
    print(f'耗时: {elapsed:.1f}s')

    if total_before > 0 or total_after > 0:
        processed = rewritten_count if rewritten_count > 0 else 1
        avg_before = total_before / processed if processed > 0 else 0
        avg_after = total_after / processed if processed > 0 else 0
        print(f'改写段落平均分: {avg_before:.1f} → {avg_after:.1f} ({avg_after - avg_before:+.1f})')

    # 保存或输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f'\n已保存到: {args.output}')
    else:
        sys.stdout.write(output_text)

    # JSON 输出
    if args.json:
        json_results = []
        for i, (para_text, sep, before_score, after_score) in enumerate(results):
            json_results.append({
                'paragraph_index': i + 1,
                'before_score': before_score,
                'after_score': after_score,
                'delta': (after_score - before_score) if before_score is not None and after_score is not None else None,
                'text_preview': para_text[:80] + '...' if len(para_text) > 80 else para_text,
            })
        json_path = (args.output or args.file).rsplit('.', 1)[0] + '_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        print(f'报告已保存到: {json_path}')

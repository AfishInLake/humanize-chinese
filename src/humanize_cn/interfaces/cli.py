#!/usr/bin/env python3
"""Unified CLI entrypoint for humanize-chinese.

Usage:
  humanize detect   <file> [options]    AI detection score (0-100)
  humanize rewrite  <file> [options]    Humanize (去 AI 味改写)
  humanize academic <file> [options]    Academic paper AIGC 降重
  humanize style    <file> --style S    风格转换
  humanize compare  <file> [options]    改写前后对比

  humanize --list                       List available subcommands
  humanize <sub> --help                 Per-subcommand help
"""
from __future__ import annotations

import argparse
import io
import json
import sys


# ─── 子命令映射 ───

SUBCOMMANDS = {
    'detect':   'AI 痕迹检测 (0-100)',
    'rewrite':  '通用去 AI 味改写',
    'academic': '学术论文 AIGC 降重（11 维度）',
    'style':    '风格转换（小红书/知乎/微博等）',
    'compare':  '改写前后对比',
}

ALIASES = {
    'humanize': 'rewrite',
    'rewrite_cn': 'rewrite',
    'acad':     'academic',
    'paper':    'academic',
    'detct':    'detect',
    'cmp':      'compare',
}

USAGE = """humanize — Chinese AI-text humanization toolkit

Usage:
  humanize <subcommand> [args]

Subcommands:
  detect     AI 痕迹检测 (0-100)
  rewrite    通用去 AI 味改写
  academic   学术论文 AIGC 降重（11 维度）
  style      风格转换（小红书/知乎/微博等）
  compare    改写前后对比

Examples:
  humanize detect 论文.txt
  humanize rewrite text.txt -o clean.txt --quick
  humanize academic 论文.txt -o 改后.txt --compare
  humanize style text.txt --style xiaohongshu -o xhs.txt
  humanize compare text.txt -a

Per-subcommand help:
  humanize detect --help
  humanize academic --help
"""


def _read_input(args):
    """Read text from file or stdin."""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f'错误: 文件未找到 {args.file}', file=sys.stderr)
            sys.exit(1)
    return sys.stdin.read()


def _save_output(text, path):
    """Save text to file or print to stdout."""
    if path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'\n结果已保存到: {path}')
    else:
        print(text)


# ─── detect ───

def _cmd_detect(argv):
    from humanize_cn.check_pkg.detect import (
        detect_patterns, calculate_score, score_to_level,
        analyze_sentences, format_output,
    )

    parser = argparse.ArgumentParser(prog='humanize detect', description='AI 痕迹检测')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('-j', '--json', action='store_true', dest='as_json')
    parser.add_argument('-s', '--score', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--sentences', type=int, default=5)
    parser.add_argument('--scene', default='general',
                        choices=['general', 'academic', 'novel', 'auto'])
    parser.add_argument('--lr', action='store_true')
    parser.add_argument('--rule-only', action='store_true')
    args = parser.parse_args(argv)

    text = _read_input(args)
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)

    issues, metrics = detect_patterns(text)
    score = calculate_score(issues, metrics)

    if args.score:
        print(score)
        return

    sentences = None
    if args.verbose or args.as_json:
        sentences = analyze_sentences(text, top_n=args.sentences)

    output = format_output(issues, metrics, score,
                           sentences=sentences,
                           as_json=args.as_json,
                           score_only=False,
                           verbose=args.verbose)
    print(output)


# ─── rewrite ───

def _cmd_rewrite(argv):
    from humanize_cn.rewrite.humanize import humanize

    parser = argparse.ArgumentParser(prog='humanize rewrite', description='通用去 AI 味改写')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--scene', default='general',
                        choices=['general', 'social', 'tech', 'formal', 'chat'])
    parser.add_argument('--style', help='写作风格')
    parser.add_argument('-a', '--aggressive', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--best-of-n', type=int, default=10)
    parser.add_argument('--no-stats', action='store_true')
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--cilin', action='store_true')
    args = parser.parse_args(argv)

    text = _read_input(args)
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)

    result = humanize(text, scene=args.scene, aggressive=args.aggressive,
                      seed=args.seed, best_of_n=args.best_of_n)
    _save_output(result, args.output)


# ─── academic ───

def _cmd_academic(argv):
    from humanize_cn.check_pkg.academic import (
        detect_academic, calculate_academic_score,
        humanize_academic, format_detect_output,
    )

    parser = argparse.ArgumentParser(prog='humanize academic', description='学术论文 AIGC 降重')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--detect-only', action='store_true')
    parser.add_argument('-a', '--aggressive', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('-j', '--json', action='store_true', dest='as_json')
    parser.add_argument('-s', '--score', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--best-of-n', type=int, default=10)
    parser.add_argument('--no-stats', action='store_true')
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args(argv)

    text = _read_input(args)
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)

    issues, metrics = detect_academic(text)
    score = calculate_academic_score(issues)

    if args.detect_only:
        output = format_detect_output(issues, metrics, score,
                                      as_json=args.as_json,
                                      score_only=args.score,
                                      verbose=args.verbose)
        print(output)
        return

    if args.compare:
        before_score = score
        result = humanize_academic(text, aggressive=args.aggressive,
                                   seed=args.seed, best_of_n=args.best_of_n)
        after_issues, after_metrics = detect_academic(result)
        after_score = calculate_academic_score(after_issues)
        print(f'改写前: {before_score}/100')
        print(f'改写后: {after_score}/100')
        print(f'降幅: {before_score - after_score}')
        _save_output(result, args.output)
        return

    result = humanize_academic(text, aggressive=args.aggressive,
                               seed=args.seed, best_of_n=args.best_of_n)
    _save_output(result, args.output)


# ─── style ───

def _cmd_style(argv):
    from humanize_cn.rewrite.style import apply_style, list_styles

    parser = argparse.ArgumentParser(prog='humanize style', description='风格转换')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('--style', required=True, help='目标风格名称')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--list', action='store_true', help='列出所有风格')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-humanize', action='store_true', help='关闭预处理')
    args = parser.parse_args(argv)

    if args.list:
        list_styles()
        return

    text = _read_input(args)
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)

    result = apply_style(text, style_name=args.style,
                         humanize_first=not args.no_humanize,
                         seed=args.seed)
    _save_output(result, args.output)


# ─── compare ───

def _cmd_compare(argv):
    from humanize_cn.check_pkg.detect import (
        detect_patterns, calculate_score, score_to_level,
    )
    from humanize_cn.rewrite.humanize import humanize

    parser = argparse.ArgumentParser(prog='humanize compare', description='改写前后对比')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('-o', '--output', help='保存改写结果')
    parser.add_argument('--scene', default='general',
                        choices=['general', 'social', 'tech', 'formal', 'chat'])
    parser.add_argument('--style', help='写作风格')
    parser.add_argument('-a', '--aggressive', action='store_true')
    args = parser.parse_args(argv)

    text = _read_input(args)
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)

    # Detect original
    print('⏳ 检测原文...')
    b_issues, b_metrics = detect_patterns(text)
    b_score = calculate_score(b_issues, b_metrics)
    b_level = score_to_level(b_score)

    # Humanize
    print('⏳ 人性化改写...')
    humanized = humanize(text, scene=args.scene, aggressive=args.aggressive)

    # Detect humanized
    print('⏳ 检测改写后...')
    a_issues, a_metrics = detect_patterns(humanized)
    a_score = calculate_score(a_issues, a_metrics)
    a_level = score_to_level(a_score)

    # Show comparison
    bar_len = 20
    b_bar = '█' * int(b_score / 100 * bar_len) + '░' * (bar_len - int(b_score / 100 * bar_len))
    a_bar = '█' * int(a_score / 100 * bar_len) + '░' * (bar_len - int(a_score / 100 * bar_len))

    print(f'\n═══ 对比结果 ═══\n')
    print(f'原文:   {b_score:3d}/100 [{b_bar}] {b_level.upper()}')
    print(f'改写后: {a_score:3d}/100 [{a_bar}] {a_level.upper()}')

    diff = b_score - a_score
    if diff > 0:
        print(f'\n✅ 降低了 {diff} 分')
    elif diff == 0:
        print(f'\n⚠️  分数未变化')
    else:
        print(f'\n❌ 分数上升了 {abs(diff)} 分')

    _save_output(humanized, args.output)


# ─── dispatcher ───

COMMANDS = {
    'detect':   _cmd_detect,
    'rewrite':  _cmd_rewrite,
    'academic': _cmd_academic,
    'style':    _cmd_style,
    'compare':  _cmd_compare,
}


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])

    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(USAGE)
        return 0

    if argv[0] in ('--list', 'list'):
        for name, desc in SUBCOMMANDS.items():
            print(f'  {name:9s} {desc}')
        return 0

    sub = ALIASES.get(argv[0], argv[0])

    if sub not in COMMANDS:
        sys.stderr.write(f'error: unknown subcommand "{argv[0]}"\n\n')
        print(USAGE, file=sys.stderr)
        return 2

    try:
        return COMMANDS[sub](argv[1:])
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())

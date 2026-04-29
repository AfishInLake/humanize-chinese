"""humanize_cn 检测模块演示

用法:
    PYTHONPATH=src python3 demo.py
    PYTHONPATH=src python3 demo.py --file 论文.txt
"""

import sys
import json

from humanize_cn import check


def print_sep(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}\n')


def fmt(v, width=8, decimals=4):
    """格式化数值，None 显示为 N/A。"""
    if v is None:
        return f'{"N/A":>{width}}'
    return f'{v:>{width}.{decimals}f}'


def demo_check(text, label='文本'):
    """演示检测功能。"""
    print_sep(f'检测: {label}')

    result = check(text)
    print(f'  检测方法:   {result["ai_method"]}')
    print(f'  AI 评分:     {result["ai_score"]}/100  ({result["ai_level"]})')
    print(f'  困惑度:     {fmt(result["perplexity"], 8, 2)}')
    print(f'  突发性:     {fmt(result["burstiness"])}')
    print(f'  平均log概率: {fmt(result["avg_log_prob"])}')
    print(f'  surprisal偏度: {fmt(result["surprisal_skew"])}')
    print(f'  surprisal峰度: {fmt(result["surprisal_kurt"])}')
    print(f'  频谱平坦度:  {fmt(result["spectral_flatness"])}')
    print(f'  top10占比:   {fmt(result["top10_ratio"])}')
    print(f'  句长CV:      {fmt(result["sentence_length_cv"])}')
    print(f'  短句占比:    {fmt(result["short_sentence_fraction"])}')
    print(f'  逗号密度:    {fmt(result["comma_density"])}')
    print(f'  字符数:      {result["char_count"]}')
    print(f'  句子数:      {result["sentence_count"]}')
    print(f'  命中规则:    {result["hit_categories"]}')

    # AI 特征标志
    flags = result['indicators']
    hit_flags = [k for k, v in flags.items() if v]
    if hit_flags:
        print(f'  统计命中:    {hit_flags}')

    return result


def demo_compare(texts):
    """对比多段文本的检测结果。"""
    print_sep('对比检测')
    print(f'  {"文本":<12} {"方法":<6} {"AI评分":>6} {"等级":<10} {"困惑度":>8} {"突发性":>8} {"句长CV":>8}')
    print(f'  {"─"*70}')

    for label, text in texts:
        r = check(text)
        print(f'  {label:<12} {r["ai_method"]:<6} {r["ai_score"]:>6} {r["ai_level"]:<10} '
              f'{fmt(r["perplexity"], 8, 2)} {fmt(r["burstiness"])} {fmt(r["sentence_length_cv"])}')


def demo_full_dict(text):
    """演示完整字典输出。"""
    print_sep('完整指标字典 (JSON)')
    result = check(text)
    output = {k: v for k, v in result.items() if k != 'issues'}
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════
#  测试文本
# ═══════════════════════════════════════════════════════════════

AI_TEXT = (
    '降AIGC不是简单降重，而是消除AI写作指纹让机器判定为人工原创。选对工具严守‘先降重后降AI’顺序，实测5秒改写5000字，维普AI率直降至0.0%——别再让无效修改耽误进度，专业方法才能省时避坑'
)

HUMAN_TEXT = (
    'AIGC，全称人工智能生成内容，简单说就是ChatGPT、文心一言等AI工具写出来的文字。这类文字有明显的机器痕迹：句式规整、用词刻板、逻辑扁平化，没有真人的思考和表达习惯，很容易被知网、维普、万方等平台的AI检测系统识别'
)

ACADEMIC_TEXT = (
    '本系统采用分层架构设计，把软件划分为三个层次：通信协议层、指令处理层和设备控制层。'
    '通信协议层负责实现UART的数据传输和帧校验功能；'
    '指令处理层利用状态机实现多设备的指令解析与分发；'
    '设备控制层则通过PWM和GPIO完成对终端硬件的驱动控制。'
)


def main():
    # 从文件读取
    if '--file' in sys.argv:
        idx = sys.argv.index('--file')
        filepath = sys.argv[idx + 1]
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        demo_check(text, filepath)
        demo_full_dict(text)
        return

    # 1. 单段检测详情
    demo_check(AI_TEXT, 'AI 文本')
    demo_check(HUMAN_TEXT, '人类文本')
    demo_check(ACADEMIC_TEXT, '学术文本')
    # 2. 多段对比
    demo_compare([
        ('AI 文本', AI_TEXT),
        ('人类文本', HUMAN_TEXT),
        ('学术文本', ACADEMIC_TEXT),
    ])

    # 3. 完整字典输出
    demo_full_dict(AI_TEXT)


if __name__ == '__main__':
    main()

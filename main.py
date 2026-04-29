#!/usr/bin/env python3
"""humanize-chinese 使用示例

运行方式:
    PYTHONPATH=src python3 main.py
"""

from humanize_cn import (
    # 检测
    detect, calculate_score, score_to_level,
    analyze_sentences,
    # 改写
    humanize,
    # 风格
    apply_style,
    # 学术
    detect_academic, calculate_academic_score, humanize_academic,
    # BERT
    bert_detect_score,
    # 语义保镖
    safe_rewrite, check_semantic_preservation,
    # 句式重组
    deep_restructure,
    # 统计
    analyze_text, compute_lr_score,
    # 配置
    load_config, get_config, get_data_dir,
)

# ============================================================
# 测试文本
# ============================================================

TEXT_SHORT = "综上所述，AI技术前景广阔。首先，我们需要提高效率。其次，应该降低成本。最后，实现可持续发展。"

TEXT_MEDIUM = """本文旨在探讨大数据技术在企业管理中的应用。通过对现有文献的梳理和分析，可以发现大数据技术已经在多个行业中得到了广泛应用。值得注意的是，数据安全和隐私保护问题也日益凸显。因此，如何在利用大数据提升企业竞争力的同时，有效保护用户隐私，是当前亟需解决的关键问题。"""

TEXT_LONG = """综上所述，人工智能技术在当今社会发挥着越来越重要的作用。首先，在医疗领域，AI技术能够辅助医生进行精准诊断，提高治疗效果。其次，在教育领域，人工智能为学生提供了个性化学习方案，促进了教育公平。最后，在交通领域，自动驾驶技术的发展将彻底改变人们的出行方式。可以说，人工智能正在深刻地改变着我们的生活方式，为社会发展注入了新的活力。"""


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. 基础检测
# ============================================================

def demo_detect():
    separator("1. AI 痕迹检测")

    text = TEXT_MEDIUM
    print(f"输入: {text[:50]}...")

    issues, metrics = detect(text)
    score = calculate_score(issues, metrics)
    level = score_to_level(score)

    print(f"AI 评分: {score}/100")
    print(f"等级: {level}")
    print(f"命中类别数: {sum(len(v) for v in issues.values())}")
    print(f"主要命中: {list(issues.keys())[:5]}")


# ============================================================
# 2. 逐句分析
# ============================================================

def demo_analyze_sentences():
    separator("2. 逐句分析（找出最可疑的句子）")

    text = TEXT_LONG
    sentences = analyze_sentences(text, top_n=3)

    for i, s in enumerate(sentences, 1):
        print(f"#{i} [分数:{s['score']}] {s['sentence'][:40]}...")
        if s.get('reasons'):
            print(f"   原因: {', '.join(s['reasons'][:3])}")


# ============================================================
# 3. 基础改写
# ============================================================

def demo_humanize():
    separator("3. 基础改写")

    text = TEXT_SHORT
    print(f"原文: {text}")

    result = humanize(text, scene='general', best_of_n=0)
    print(f"改写: {result}")

    # 改写前后对比
    before = calculate_score(*detect(text))
    after = calculate_score(*detect(result))
    print(f"分数: {before} → {after} (降幅 {before - after})")


# ============================================================
# 4. 多场景改写
# ============================================================

def demo_scenes():
    separator("4. 多场景改写")

    text = "通过技术手段，我们能够有效提高工作效率。"

    for scene in ['general', 'formal', 'social', 'chat']:
        result = humanize(text, scene=scene, best_of_n=0)
        print(f"[{scene:8s}] {result}")


# ============================================================
# 5. 激进模式
# ============================================================

def demo_aggressive():
    separator("5. 普通 vs 激进模式")

    text = TEXT_MEDIUM

    normal = humanize(text, aggressive=False, best_of_n=0)
    aggressive = humanize(text, aggressive=True, best_of_n=0)

    n_score = calculate_score(*detect(normal))
    a_score = calculate_score(*detect(aggressive))

    print(f"原文:    {calculate_score(*detect(text))}/100")
    print(f"普通:    {n_score}/100")
    print(f"激进:    {a_score}/100")
    print(f"\n普通改写: {normal[:60]}...")
    print(f"激进改写: {aggressive[:60]}...")


# ============================================================
# 6. 风格转换
# ============================================================

def demo_style():
    separator("6. 风格转换")

    text = "今天去了一家新开的咖啡店，拿铁很好喝，环境也不错。"

    for style in ['casual', 'xiaohongshu', 'zhihu', 'weibo']:
        result = apply_style(text, style_name=style, humanize_first=False)
        print(f"[{style:12s}] {result[:50]}...")


# ============================================================
# 7. 学术降重
# ============================================================

def demo_academic():
    separator("7. 学术降重")

    text = TEXT_MEDIUM
    print(f"原文: {text[:50]}...")

    issues, metrics = detect_academic(text)
    before_score = calculate_academic_score(issues)
    print(f"学术 AI 评分: {before_score}/100")

    result = humanize_academic(text, aggressive=False, best_of_n=0)
    after_issues, _ = detect_academic(result)
    after_score = calculate_academic_score(after_issues)

    print(f"降重后评分: {after_score}/100 (降幅 {before_score - after_score})")
    print(f"改写: {result[:60]}...")


# ============================================================
# 8. BERT 检测（需要模型文件）
# ============================================================

def demo_bert():
    separator("8. BERT 检测")

    text = TEXT_MEDIUM
    score = bert_detect_score(text)

    if score is not None:
        print(f"BERT AI 概率: {score}/100")
    else:
        print("BERT 模型不可用（未安装模型文件）")
        print("提示: 将模型文件放入 src/humanize_cn/data/models/detector/")


# ============================================================
# 9. 语义保镖（需要模型文件）
# ============================================================

def demo_semantic_guard():
    separator("9. 语义保镖")

    original = "通过技术手段提高效率"
    rewritten_good = "靠技术手段提升了效率"
    rewritten_bad = "昨天下了一场大雨"

    for rw in [rewritten_good, rewritten_bad]:
        ok, sim = check_semantic_preservation(original, rw, threshold=0.85)
        result = safe_rewrite(original, rw, threshold=0.85)
        status = "✅ 保持" if ok else "❌ 偏移"
        print(f"原文: {original}")
        print(f"改写: {rw}")
        print(f"相似度: {sim:.4f} {status}")
        if result != rw:
            print(f"→ 回退为: {result}")
        print()


# ============================================================
# 10. 句式重组
# ============================================================

def demo_restructure():
    separator("10. 句式重组")

    text = "首先，我们需要提高效率。其次，应该降低成本。最后，实现可持续发展。"
    print(f"原文: {text}")

    result = deep_restructure(text, aggressive=False, scene='general')
    print(f"重组: {result}")


# ============================================================
# 11. 统计分析
# ============================================================

def demo_stats():
    separator("11. 统计分析")

    text = TEXT_LONG
    analysis = analyze_text(text)

    ppl = analysis.get('perplexity', 'N/A')
    if isinstance(ppl, dict):
        ppl = ppl.get('perplexity', 'N/A')
    burst = analysis.get('burstiness', 'N/A')
    if isinstance(burst, dict):
        burst = burst.get('cv', 'N/A')
    mattr = analysis.get('char_mattr', 'N/A')
    trans = analysis.get('transition_density', 'N/A')
    if isinstance(trans, dict):
        trans = trans.get('per_1k_chars', 'N/A')

    print(f"困惑度: {ppl:.1f}" if isinstance(ppl, (int, float)) else f"困惑度: {ppl}")
    print(f"突发度: {burst:.3f}" if isinstance(burst, (int, float)) else f"突发度: {burst}")
    print(f"词汇多样性 (MATTR): {mattr:.3f}" if isinstance(mattr, (int, float)) else f"词汇多样性: {mattr}")
    print(f"过渡词密度: {trans:.1f}/千字" if isinstance(trans, (int, float)) else f"过渡词密度: {trans}")

    # LR 评分
    lr = compute_lr_score(text, scene='general')
    if lr:
        print(f"LR AI 概率: {lr['score']:.1f}/100")


# ============================================================
# 12. 配置查看
# ============================================================

def demo_config():
    separator("12. 配置系统")

    config = load_config()
    print(f"数据目录: {get_data_dir()}")
    print(f"检测 BERT 权重: {get_config('detect', 'fuse_bert_weight', 'N/A')}")
    print(f"改写 best_of_n: {get_config('global', 'default_best_of_n', 'N/A')}")
    print(f"BERT 检测器启用: {get_config('bert_detector', 'enabled', 'N/A')}")
    print(f"语义保镖阈值: {get_config('semantic_guard', 'similarity_threshold', 'N/A')}")


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys

    demos = {
        'all':   [demo_detect, demo_analyze_sentences, demo_humanize,
                  demo_scenes, demo_aggressive, demo_style, demo_academic,
                  demo_bert, demo_semantic_guard, demo_restructure,
                  demo_stats, demo_config],
        '1':     [demo_detect],
        '2':     [demo_analyze_sentences],
        '3':     [demo_humanize],
        '4':     [demo_scenes],
        '5':     [demo_aggressive],
        '6':     [demo_style],
        '7':     [demo_academic],
        '8':     [demo_bert],
        '9':     [demo_semantic_guard],
        '10':    [demo_restructure],
        '11':    [demo_stats],
        '12':    [demo_config],
    }

    choice = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if choice not in demos:
        print("用法: python main.py [编号]")
        print("  all  - 运行全部示例（默认）")
        for k in sorted(demos.keys(), key=lambda x: (0 if x == 'all' else int(x))):
            print(f"  {k:>4s} - 运行示例 {k}")
        sys.exit(1)

    for fn in demos[choice]:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ 错误: {e}")

    print(f"\n{'='*60}")
    print("  示例运行完毕")
    print(f"{'='*60}")

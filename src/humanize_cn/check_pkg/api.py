"""统一检测模块 — 输入文本，输出完整指标字典。

将检测和改写彻底解耦。本模块只负责检测，不涉及任何改写逻辑。

检测策略优先级：
1. BERT + Ensemble 融合（如果两者都可用）→ 三路融合评分
2. BERT 序列分类模型（如果可用）→ 直接输出 0~100 的 AI 概率
3. Ensemble 评分器（XGBoost/LR）→ 融合评分
4. 规则检测 + n-gram 统计特征（兜底）→ 综合评分

使用方式：
    from humanize_cn.check_pkg.api import check

    result = check("这是一段待检测的文本。")
    print(result['ai_score'])          # AI 概率 0-100
    print(result['ai_method'])         # 'bert', 'ensemble', 'xgboost', 'lr', 'ngram'
    print(result['perplexity'])        # 困惑度
    print(result['burstiness'])        # 突发性
    print(result['indicators'])        # 各指标布尔标志
"""

from ..models.ngram import analyze_text, compute_perplexity
from .detect import detect_patterns, calculate_score
from loguru import logger


def _try_bert_score(text):
    """尝试用 BERT 模型计算 AI 概率。

    Returns:
        float or None: 0-100 的 AI 概率，模型不可用时返回 None
    """
    try:
        from ..models.bert_detector import bert_detect_score
        return bert_detect_score(text)
    except Exception as e:
        logger.warning("BERT 检测失败: {}", e)
        return None


def _try_ensemble_score(text, scene='auto'):
    """尝试用 Ensemble (XGBoost/LR) 评分。

    Returns:
        dict or None: {'p_ai': float, 'score': int, 'method': str}
    """
    try:
        from ..models.ensemble_scorer import ensemble_score
        return ensemble_score(text, scene=scene)
    except Exception:
        return None


def _fuse_scores(bert_score, ensemble_result, rule_score):
    """自适应双路融合: BERT(主) + XGBoost(辅)。

    策略:
    - BERT 非常自信（≥90 或 ≤10）时，直接信任 BERT，XGBoost 不参与
    - BERT 不确定（10-90）时，BERT 0.7 + XGBoost 0.3 融合
    - 只有一路可用时，直接使用该路分数

    Args:
        bert_score: float 0-100 or None
        ensemble_result: dict with 'score' key, or None
        rule_score: int 0-100 (unused, kept for API compatibility)

    Returns:
        int: fused AI score 0-100
    """
    if bert_score is not None and ensemble_result is not None:
        if bert_score >= 90 or bert_score <= 10:
            # BERT 非常自信，直接采用
            score = bert_score
        else:
            # BERT 不确定，引入 XGBoost 辅助
            score = 0.7 * bert_score + 0.3 * ensemble_result['score']
    elif bert_score is not None:
        score = bert_score
    elif ensemble_result is not None:
        score = ensemble_result['score']
    else:
        score = rule_score

    return int(round(min(100, max(0, score))))


def check(text):
    """对文本进行 AI 痕迹检测，返回完整指标字典。

    优先使用 BERT 序列分类模型（输出 0~100 的 AI 概率），
    模型不可用时自动降级到规则检测 + n-gram 统计特征。

    Args:
        text: 待检测的中文文本，不限长度
    Returns:
        dict: 包含以下字段的指标字典

        核心指标：
        - ai_score (int):          AI 概率评分 0-100，越高越可能是 AI 生成
        - ai_level (str):          评级 LOW / MEDIUM / HIGH / VERY HIGH
        - ai_method (str):         检测方法 'bert' 或 'ngram'
        - perplexity (float):      字符级困惑度
        - burstiness (float):      窗口困惑度变异系数（突发性）
        - avg_log_prob (float):    每字符平均 log2 概率

        DivEye 特征：
        - surprisal_skew (float):  困惑度偏度
        - surprisal_kurt (float):  困惑度峰度（超额）
        - spectral_flatness (float): 频谱平坦度

        GLTR 特征：
        - top10_ratio (float):     高概率字符占比（top-10 bucket）
        - top100_ratio (float):    top-100 bucket 占比

        句长特征：
        - sentence_count (int):    句子数
        - mean_sentence_len (float): 平均句长（中文字符）
        - sentence_length_cv (float): 句长变异系数
        - short_sentence_fraction (float): 短句占比（<10字）
        - long_sentence_fraction (float):  长句占比（>30字）

        标点特征：
        - comma_density (float):   逗号密度（每100非空白字符）

        文本基础：
        - char_count (int):        中文字符数
        - entropy (float | None):  字符 bigram 熵

        AI 特征标志（布尔）：
        - indicators (dict):       各指标是否命中 AI 特征

        规则检测详情：
        - issues (dict):           命中的规则类别及详情
        - hit_categories (list):   命中的规则类别名称列表
    """
    # 1. 尝试 BERT 模型检测 (不变)
    bert_score = _try_bert_score(text)

    # 2. 尝试 Ensemble (XGBoost/LR) 评分 (新增)
    ensemble_result = _try_ensemble_score(text)

    # 3. 统计特征分析（始终计算，BERT 模式下作为辅助参考）
    stats = analyze_text(text)
    ppl_result = compute_perplexity(text, window_size=0)

    # 4. 确定 AI 评分和方法
    if bert_score is not None and ensemble_result is not None:
        # 三路融合: BERT + Ensemble + Rule
        issues, metrics = detect_patterns(text)
        rule_score = calculate_score(issues, metrics)
        fused = _fuse_scores(bert_score, ensemble_result, rule_score)
        ai_score = fused
        ai_method = 'ensemble'
        issues = {}
        hit_categories = []
    elif bert_score is not None:
        ai_score = int(round(bert_score))
        ai_method = 'bert'
        issues = {}
        hit_categories = []
    elif ensemble_result is not None:
        ai_score = ensemble_result['score']
        ai_method = ensemble_result['method']  # 'xgboost' or 'lr'
        issues = {}
        hit_categories = []
    else:
        # 最终兜底: 规则+统计 (不变)
        ai_method = 'ngram'
        issues, metrics = detect_patterns(text)
        ai_score = calculate_score(issues, metrics)

        hit_categories = []
        for category, items in issues.items():
            if items and not category.startswith('stat_'):
                hit_categories.append(category)

    # 5. 评级
    if ai_score <= 24:
        ai_level = 'LOW'
    elif ai_score <= 49:
        ai_level = 'MEDIUM'
    elif ai_score <= 74:
        ai_level = 'HIGH'
    else:
        ai_level = 'VERY HIGH'

    # ── 组装输出变量 ──
    diveye = stats.get('diveye', {})
    gltr = stats.get('gltr', {})
    gltr_props = gltr.get('proportions', {}) if gltr else {}
    sent_len = stats.get('sent_len', {})
    punct = stats.get('punct', {})
    indicators = stats.get('indicators', {})

    # ── 句长特征（句子数 < 3 时不可靠，返回 None）──
    _sent_reliable = sent_len.get('mean_len', 0) > 0
    # ── 统计特征（字符数 < 30 时不可靠）──
    _stat_reliable = stats.get('char_count', 0) >= 30

    return {
        # ── 核心指标 ──
        'ai_score': ai_score,
        'ai_level': ai_level,
        'ai_method': ai_method,
        'perplexity': stats.get('perplexity', 0) if _stat_reliable else None,
        'burstiness': stats.get('burstiness', 0) if _stat_reliable else None,
        'avg_log_prob': ppl_result.get('avg_log_prob', 0) if _stat_reliable else None,

        # ── DivEye 特征 ──
        'surprisal_skew': diveye.get('skew', 0) if _stat_reliable else None,
        'surprisal_kurt': diveye.get('excess_kurt', 0) if _stat_reliable else None,
        'spectral_flatness': diveye.get('spectral_flatness', 0) if _stat_reliable else None,

        # ── GLTR 特征 ──
        'top10_ratio': gltr_props.get('top10', 0) if _stat_reliable else None,
        'top100_ratio': gltr_props.get('top100', 0) if _stat_reliable else None,

        # ── 句长特征 ──
        'sentence_count': sent_len.get('n_sentences', 0),
        'mean_sentence_len': sent_len.get('mean_len', 0) if _sent_reliable else None,
        'sentence_length_cv': sent_len.get('cv', 0) if _sent_reliable else None,
        'short_sentence_fraction': sent_len.get('short_frac', 0) if _sent_reliable else None,
        'long_sentence_fraction': sent_len.get('long_frac', 0) if _sent_reliable else None,

        # ── 标点特征 ──
        'comma_density': punct.get('comma_density', 0),

        # ── 文本基础 ──
        'char_count': stats.get('char_count', 0),
        'entropy': None,  # 仅 ngram 模式下由 detect_patterns 计算

        # ── 其他统计 ──
        'entropy_cv': stats.get('entropy_cv', 0) if _stat_reliable else None,
        'char_mattr': stats.get('char_mattr', 0) if _stat_reliable else None,
        'uni_ppl': stats.get('uni_ppl', 0) if _stat_reliable else None,
        'uni_tri_ratio': stats.get('uni_tri_ratio', 0) if _stat_reliable else None,

        # ── AI 特征标志 ──
        'indicators': indicators,

        # ── 规则检测详情（仅 ngram 模式）──
        'issues': dict(issues) if bert_score is None else {},
        'hit_categories': hit_categories,
    }

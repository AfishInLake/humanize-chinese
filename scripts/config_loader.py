#!/usr/bin/env python3
"""
统一配置加载器。

所有脚本共享同一个配置文件 config.json，
通过 load_config() 获取配置，支持默认值和用户覆盖。

用法:
    from config_loader import load_config
    cfg = load_config()
    threshold = cfg['humanize']['bigram_strength']
"""

import os
import json
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.json')
_USER_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.user.json')

# ═══════════════════════════════════════════════════════════════════
#  默认配置（与原代码硬编码值一致）
# ═══════════════════════════════════════════════════════════════════

DEFAULTS = {
    # ─── 全局开关 ───
    "global": {
        "use_noise": True,
        "use_cilin": False,
        "use_stats": True,
        "default_best_of_n": 10,
        "seed": 42
    },

    # ─── humanize_cn.py 改写参数 ───
    "humanize": {
        "tier_thresholds": {
            "full": 25,
            "moderate": 5
        },
        "aggressive_casualness_boost": 0.3,
        "bigram_strength": 0.3,
        "bigram_strength_aggressive": 0.5,
        "bigram_strength_moderate_factor": 0.6,
        "bigram_replace_ratio": 0.6,
        "bigram_candidate_position": "lower_third",
        "noise_density": 0.15,
        "noise_density_aggressive": 0.25,
        "transition_density_target": 6.0,
        "transition_density_target_long": 3.0,
        "long_text_threshold": 1500,
        "ppl_repair_range": [0, 350],
        "ppl_repair_max_sentences": 5,
        "ppl_repair_ratio": 0.2,
        "burstiness_retreat_threshold": 0.8,
        "merge_short_min_len": 8,
        "split_long_max_len": 80,
        "dialogue_density_threshold": 0.08,
        "noise_min_sentence_cn": 8,
        "paragraph_merge_factor": 0.6,
        "paragraph_split_factor": 1.5,
        "paragraph_merge_prob": 0.4,
        "comma_density_target": 4.7,
        "sentence_particle_rate": 0.15,
        "sentence_particle_cn_range": [6, 40],
        "casual_min_casualness": 0.2,
        "casual_inject_rate_factor": 0.2,
        "shorten_max_length": 150,
        "diversify_repeat_threshold": 2,
        "zipf_freq_range": [2, 4],
        "zipf_boost_count_min": 2,
        "zipf_boost_count_max": 5,
        "randomize_merge_rate": 0.15,
        "randomize_merge_rate_aggressive": 0.25,
        "randomize_truncate_rate": 0.15,
        "randomize_truncate_rate_aggressive": 0.25,
        "randomize_merge_max_cn": 100,
        "randomize_truncate_cn_range": [20, 50],
        "scenes": {
            "general": {"casualness": 0.3},
            "social": {"casualness": 0.7},
            "tech": {"casualness": 0.3},
            "formal": {"casualness": 0.1},
            "chat": {"casualness": 0.8},
            "academic": {"casualness": 0.1},
            "novel": {"casualness": 0.2}
        }
    },

    # ─── restructure_cn.py 句式重组参数 ───
    "restructure": {
        "deep_strength_normal": 0.4,
        "deep_strength_aggressive": 0.6,
        "deep_delete_prob_normal": 0.35,
        "deep_delete_prob_aggressive": 0.6,
        "deep_comma_density_target": 5.0,
        "sentence_strength": 0.6,
        "sentence_min_cn": 10,
        "sentence_length_tolerance": 0.5,
        "sentence_min_result_len": 4,
        "dep_restructure_cn_range": [10, 60],
        "dep_restructure_min_result_cn": 8,
        "dep_split_min_cn": 30,
        "dep_split_min_half_cn": 4,
        "dep_split_prob": 0.5,
        "split_min_cn": 25,
        "split_bujin_prob": 0.5,
        "split_connector_min_cn": 30,
        "split_connector_prob": 0.4,
        "merge_min_parts": 5,
        "merge_short_frac_threshold": 0.20,
        "merge_short_cn_threshold": 10,
        "merge_short_max_single_cn": 20,
        "merge_short_max_combined_cn": 45,
        "merge_short_prob": 0.4,
        "merge_shared_subject_range": [2, 6],
        "merge_shared_min_len": 2,
        "reorder_min_sentences": 4,
        "reorder_swap_prob": 0.5,
        "filler_delete_prob": 0.5,
        "reaction_target_short_frac_academic": 0.22,
        "reaction_target_short_frac_general": 0.15,
        "reaction_max_per_paragraph": 1,
        "reaction_min_sentences": 3,
        "reaction_social_prob": 0.35,
        "reaction_tail_min_cn": 15,
        "comma_density_target": 4.7,
        "comma_min_text_len": 100,
        "comma_min_sentence_cn": 15,
        "comma_max_existing": 2,
        "comma_prefix_min_cn": 6,
        "comma_suffix_min_cn": 4,
        "diversify_target_cv": 0.42,
        "diversify_target_short_frac": 0.10,
        "diversify_split_max_iter": 3,
        "diversify_split_min_cn": 40,
        "diversify_split_min_commas": 3,
        "diversify_split_min_half_cn": 4,
        "dialogue_density_threshold": 0.08,
        "sentence_stats_min_cn": 3,
        "sentence_stats_min_sentences": 3
    },

    # ─── detect_cn.py 检测参数 ───
    "detect": {
        "short_text_threshold": 100,
        "fuse_rule_weight": 0.2,
        "fuse_lr_weight": 0.8,
        "severity_weights": {
            "critical": 8,
            "high": 4,
            "medium": 2,
            "style": 1.5,
            "statistical": 0
        },
        "rule_score_cap": 60,
        "stat_score_cap": 40,
        "repeat_decay_max": 5,
        "emotional_density_penalty_threshold": 0.1,
        "emotional_density_penalty_min_chars": 500,
        "emotional_density_penalty_score": 5,
        "low_entropy_threshold": 5.5,
        "low_entropy_penalty_score": 5,
        "score_levels": {
            "very_high": 75,
            "high": 50,
            "medium": 25
        },
        "analyze_phrase_score": 3,
        "analyze_template_score": 5,
        "analyze_default_top_n": 5,
        "format_bar_length": 20,
        "format_default_display": 3,
        "format_verbose_display": 5,
        "main_default_sentences": 5,
        "char_entropy_min_chars": 10,
        "char_entropy_default": 5.0,
        "hedging_threshold": 5,
        "punctuation_dash_threshold": 1.0,
        "punctuation_semicolon_threshold": 0.5,
        "rhetoric_threshold": 2,
        "uniform_paragraphs_cv": 0.2,
        "uniform_paragraphs_min": 3,
        "low_burstiness_cv": 0.25,
        "low_burstiness_min_sentences": 5,
        "emotional_flatness_min_chars": 300,
        "emotional_flatness_density": 0.15,
        "repetitive_starters_min_sentences": 5,
        "repetitive_starters_threshold": 3,
        "low_entropy_min_chars": 200,
        "low_entropy_threshold": 6.0,
        "ngram_min_chars": 100
    },

    # ─── ngram_model.py 统计模型参数 ───
    "ngram": {
        "vocab_min": 1000,
        "smoothing_k": 0.01,
        "log_prob_floor": -20.0,
        "trigram_weight": 0.6,
        "perplexity_min_chars": 5,
        "burstiness_window_size": 50,
        "spectral_flatness_min_seq": 16,
        "spectral_flatness_max_sample": 256,
        "spectral_flatness_power_floor": 1e-12,
        "distribution_moments_min_seq": 4,
        "gltr_min_chars": 30,
        "gltr_bucket_boundaries": [10, 100, 1000],
        "diveye_min_seq": 16,
        "curvature_n_positions": 50,
        "curvature_k_alts": 10,
        "curvature_min_chars": 30,
        "top_chars_k": 500,
        "news_lp_diff_min_chars": 30,
        "wiki_lp_diff_min_chars": 30,
        "binoculars_min_chars": 30,
        "transition_density_min_cn": 50,
        "sentence_length_min_cn": 3,
        "short_sentence_threshold": 10,
        "long_sentence_threshold": 30,
        "ai_equal_range": [15, 25],
        "sentence_length_min_sentences": 3,
        "mattr_window": 100,
        "burstiness_min_windows": 3,
        "entropy_uniformity_min_paragraphs": 3,
        "entropy_uniformity_min_para_cn": 20,
        "entropy_uniformity_min_sent_chars": 10,
        "analyze_min_chars": 30,
        "auto_scene_short_threshold": 1500,
        "logit_clamp": [-500, 500],
        "indicators": {
            "low_perplexity": {"ppl_range": [50, 500], "min_chars": 200},
            "low_burstiness": {"burst_threshold": 0.12, "min_windows": 6},
            "uniform_entropy": {"cv_threshold": 0.05, "min_paragraphs": 3},
            "low_surprisal_skew": {"skew_threshold": 1.35, "min_chars": 150},
            "low_surprisal_kurt": {"kurt_threshold": 0.35, "min_chars": 150},
            "high_top10_bucket": {"top10_threshold": 0.21, "min_chars": 150},
            "low_sentence_length_cv": {"cv_threshold": 0.40, "min_sentences": 5},
            "low_short_sentence_fraction": {"frac_threshold": 0.08, "min_sentences": 5},
            "low_comma_density": {"density_threshold": 4.5, "min_chars": 100}
        }
    },

    # ─── academic_cn.py 学术检测参数 ───
    "academic": {
        "use_stats": True,
        "use_noise": True,
        "default_best_of_n": 10,
        "rule_score_cap": 60,
        "rule_score_multiplier": 0.7,
        "stat_score_cap": 40,
        "severity_multipliers": {
            "critical": 1.5,
            "high": 1.0,
            "medium": 0.6,
            "low": 0.3
        },
        "score_levels": {
            "very_high": 75,
            "high": 50,
            "medium": 25
        },
        "paragraph_uniformity_cv": 0.18,
        "paragraph_opener_repeat_threshold": 3,
        "connector_density_threshold": 3.0,
        "synonym_poverty_repeat_threshold": 4,
        "synonym_poverty_display_limit": 5,
        "citation_min_count": 3,
        "citation_template_ratio": 0.6,
        "perfect_conclusion_threshold": 2,
        "perfect_conclusion_tail_start": 0.8,
        "certainty_min_count": 3,
        "certainty_max_hedge": 2,
        "topic_diffusion_min_paragraphs": 3,
        "topic_diffusion_min_chars": 300,
        "topic_diffusion_threshold": 0.5,
        "topic_diffusion_min_para_chars": 20,
        "topic_diffusion_signature_size": 20,
        "hedging_inject_max_normal": "max(3, total_sents // 5)",
        "hedging_inject_max_aggressive": "max(3, total_sents // 4)",
        "hedging_inject_prob_normal": 0.14,
        "hedging_inject_prob_aggressive": 0.15,
        "hedging_inject_min_prefix_cn": 4,
        "author_voice_max_normal": 5,
        "author_voice_max_aggressive": 6,
        "break_uniform_merge_cn": 60,
        "break_uniform_merge_prob": 0.25,
        "break_uniform_swap_prob": 0.15,
        "reduce_connectors_max_normal": 6,
        "reduce_connectors_max_aggressive": 7,
        "reduce_connectors_prob": 0.5,
        "shorten_long_max_chars": 90,
        "bigram_strength": 0.45,
        "bigram_strength_aggressive": 0.5,
        "noise_density": 0.18,
        "noise_density_aggressive": 0.2,
        "ppl_repair_range": [0, 350],
        "ppl_repair_max_sentences": 3,
        "format_bar_length": 20,
        "format_default_display": 4,
        "format_verbose_display": 8
    },

    # ─── style_cn.py 风格转换参数 ───
    "style": {
        "emoji_density": 0.2,
        "shorten_max_length": 120,
        "casual_tone_prob": 0.15,
        "casual_connector_prob": 0.15,
        "casual_emoji_density": 0.1,
        "zhihu_example_prob": 0.3,
        "zhihu_support_prob": 0.1,
        "zhihu_ending_prob": 0.3,
        "xhs_word_replace_prob": 0.3,
        "xhs_exclaim_prob": 0.3,
        "xhs_emoji_density": 0.4,
        "xhs_max_para_length": 80,
        "xhs_tag_count": 3,
        "wechat_rhetorical_prob": 0.4,
        "wechat_ending_prob": 0.4,
        "academic_exclaim_limit": 2,
        "literary_imagery_prob": 0.3,
        "literary_imagery_max": 2,
        "literary_sensory_prob": 0.15,
        "novel_meta_delete_threshold": 300,
        "novel_artifact_check_len": 60,
        "weibo_max_sentences": 5,
        "weibo_attitude_prob": 0.4,
        "weibo_emoji_density": 0.15
    },

    # ─── compare_cn.py 对比参数 ───
    "compare": {
        "detect_timeout": 30,
        "humanize_timeout": 30,
        "format_bar_length": 20,
        "default_scene": "general"
    },

    # ─── semantic_guard.py 语义检查参数 ───
    "semantic_guard": {
        "onnx_model_path": "bert_base_chinese.onnx",
        "similarity_threshold": 0.85,
        "max_token_length": 512
    }
}


# ═══════════════════════════════════════════════════════════════════
#  配置加载
# ═══════════════════════════════════════════════════════════════════

_config_cache = None


def _deep_merge(base, override):
    """深度合并两个字典，override 中的值覆盖 base。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path=None):
    """加载配置文件，返回合并后的配置字典。

    加载顺序（后者覆盖前者）:
      1. 内置默认值 DEFAULTS
      2. config.json（项目默认配置）
      3. config.user.json（用户自定义配置，可选）

    Args:
        config_path: 自定义配置文件路径（可选）

    Returns:
        dict: 合并后的配置字典
    """
    global _config_cache

    if _config_cache is not None and config_path is None:
        return _config_cache

    # 从默认值开始
    config = copy.deepcopy(DEFAULTS)

    # 加载 config.json
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            config = _deep_merge(config, file_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f'[config] 警告: 无法加载 {CONFIG_FILE}: {e}', file=__import__('sys').stderr)

    # 加载 config.user.json（用户覆盖，不提交到 git）
    if os.path.exists(_USER_CONFIG_FILE):
        try:
            with open(_USER_CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            config = _deep_merge(config, user_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f'[config] 警告: 无法加载 {_USER_CONFIG_FILE}: {e}', file=__import__('sys').stderr)

    # 加载自定义路径
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
            config = _deep_merge(config, custom_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f'[config] 警告: 无法加载 {config_path}: {e}', file=__import__('sys').stderr)

    if config_path is None:
        _config_cache = config

    return config


def get_config(section, key=None, default=None):
    """快捷获取配置值。

    Args:
        section: 配置分区名（如 'humanize', 'detect'）
        key: 配置键名（如 'bigram_strength'），为 None 时返回整个分区
        default: 默认值（键不存在时返回）

    Returns:
        配置值或整个分区字典
    """
    cfg = load_config()
    if section not in cfg:
        return default
    if key is None:
        return cfg[section]
    return cfg[section].get(key, default)


def reload_config():
    """清除缓存，强制重新加载配置。"""
    global _config_cache
    _config_cache = None
    return load_config()


def save_user_config(config_dict):
    """保存用户配置到 config.user.json。

    只保存与默认值不同的部分。

    Args:
        config_dict: 用户配置字典
    """
    defaults = copy.deepcopy(DEFAULTS)
    diff = _extract_diff(defaults, config_dict)
    with open(_USER_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(diff, f, ensure_ascii=False, indent=2)
    reload_config()


def _extract_diff(defaults, config):
    """提取与默认值不同的配置项。"""
    diff = {}
    for key, value in config.items():
        if key not in defaults:
            diff[key] = value
        elif isinstance(defaults[key], dict) and isinstance(value, dict):
            sub_diff = _extract_diff(defaults[key], value)
            if sub_diff:
                diff[key] = sub_diff
        elif defaults[key] != value:
            diff[key] = value
    return diff


if __name__ == '__main__':
    import sys
    cfg = load_config()
    print(f'配置加载成功，共 {len(cfg)} 个分区:')
    for section in cfg:
        print(f'  {section}: {len(cfg[section])} 项')
    print(f'\n用户配置文件: {_USER_CONFIG_FILE}')
    print(f'项目配置文件: {CONFIG_FILE}')

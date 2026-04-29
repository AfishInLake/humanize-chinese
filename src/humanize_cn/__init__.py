"""humanize_cn — 中文 AI 文本检测与改写工具包

两个核心模块：
    from humanize_cn.check_pkg.api import check   # 检测
    from humanize_cn.rewrite.humanize import humanize  # 改写
"""

import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

import logging
logging.getLogger('jieba').setLevel(logging.WARNING)

__version__ = '0.1.0'

# ═══════════════════════════════════════════════════════════════
#  检测模块（check）
# ═══════════════════════════════════════════════════════════════
from .check_pkg.api import check

from .check_pkg.detect import (
    detect_patterns as detect,
    calculate_score,
    score_to_level,
    analyze_sentences,
    format_output,
    split_sentences,
    count_chinese_chars, detect_patterns,
)

from .check_pkg.academic import (
    detect_academic,
    calculate_academic_score,
    humanize_academic,
    topic_diffusion,
)

# ═══════════════════════════════════════════════════════════════
#  改写模块（rewrite）
# ═══════════════════════════════════════════════════════════════
from .rewrite.humanize import (
    humanize,
    remove_three_part_structure,
    replace_phrases,
    reduce_high_freq_bigrams,
    inject_noise_expressions,
    randomize_sentence_lengths,
    merge_short_sentences,
    split_long_sentences,
    cap_transition_density,
    diversify_vocabulary,
    zipf_perturb,
    vary_paragraph_rhythm,
    reduce_punctuation,
    add_casual_expressions,
    shorten_paragraphs,
)

from .rewrite.style import (
    apply_style,
    list_styles,
)

from .rewrite.paragraph import humanize_paragraph as paragraph_humanize

# ═══════════════════════════════════════════════════════════════
#  模型 & 配置（内部依赖，按需导入）
# ═══════════════════════════════════════════════════════════════
from .models.bert_detector import (
    bert_detect_score,
    bert_detect_batch,
    get_bert_score,
)

from .models.semantic_guard import (
    check_semantic_preservation,
    safe_rewrite,
    cosine_similarity,
)

from .models.restructure import (
    deep_restructure,
    restructure_sentences,
    bert_naturalness_score,
)

from .models.ngram import (
    analyze_text,
    compute_lr_score,
    compute_perplexity,
)

from .config import (
    load_config,
    get_config,
    reload_config,
    save_user_config,
    get_data_dir,
)

__all__ = [
    # ── 检测 ──
    'check',
    'detect', 'calculate_score', 'score_to_level',
    'analyze_sentences', 'format_output', 'split_sentences', 'count_chinese_chars',
    'detect_academic', 'calculate_academic_score', 'humanize_academic', 'topic_diffusion',
    # ── 改写 ──
    'humanize', 'remove_three_part_structure', 'replace_phrases',
    'reduce_high_freq_bigrams', 'inject_noise_expressions',
    'randomize_sentence_lengths', 'merge_short_sentences', 'split_long_sentences',
    'cap_transition_density', 'diversify_vocabulary', 'zipf_perturb',
    'vary_paragraph_rhythm', 'reduce_punctuation', 'add_casual_expressions',
    'shorten_paragraphs', 'apply_style', 'list_styles', 'paragraph_humanize',
    # ── 模型 ──
    'bert_detect_score', 'bert_detect_batch', 'get_bert_score',
    'check_semantic_preservation', 'safe_rewrite', 'cosine_similarity',
    'deep_restructure', 'restructure_sentences', 'bert_naturalness_score',
    'analyze_text', 'compute_lr_score', 'compute_perplexity',
    # ── 配置 ──
    'load_config', 'get_config', 'reload_config', 'save_user_config', 'get_data_dir',
]

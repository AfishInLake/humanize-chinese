"""humanize_cn — 中文 AI 文本去痕迹工具包

快速开始:
    from humanize_cn import detect, humanize, score_to_level

    issues, metrics = detect("综上所述，AI技术前景广阔。")
    from humanize_cn.detection.detect import calculate_score
    score = calculate_score(issues, metrics)
    print(f"AI 评分: {score}, 等级: {score_to_level(score)}")

    from humanize_cn.rewriting.humanize import humanize as rewrite
    result = rewrite("综上所述，AI技术前景广阔。")
    print(result)
"""

__version__ = '0.1.0'

# ─── 便捷顶层导入 ───
from .detection.detect import (
    detect_patterns as detect,
    calculate_score,
    score_to_level,
    analyze_sentences,
    format_output,
    split_sentences,
    count_chinese_chars,
)

from .rewriting.humanize import (
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

from .rewriting.style import (
    apply_style,
    list_styles,
)

from .rewriting.paragraph import humanize_paragraph as paragraph_humanize

from .detection.academic import (
    detect_academic,
    calculate_academic_score,
    humanize_academic,
    topic_diffusion,
)

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
    # detection
    'detect', 'detect_patterns', 'calculate_score', 'score_to_level',
    'analyze_sentences', 'format_output', 'split_sentences', 'count_chinese_chars',
    'detect_academic', 'calculate_academic_score', 'humanize_academic', 'topic_diffusion',
    # rewriting
    'humanize', 'remove_three_part_structure', 'replace_phrases',
    'reduce_high_freq_bigrams', 'inject_noise_expressions',
    'randomize_sentence_lengths', 'merge_short_sentences', 'split_long_sentences',
    'cap_transition_density', 'diversify_vocabulary', 'zipf_perturb',
    'vary_paragraph_rhythm', 'reduce_punctuation', 'add_casual_expressions',
    'shorten_paragraphs', 'apply_style', 'list_styles', 'paragraph_humanize',
    # models
    'bert_detect_score', 'bert_detect_batch', 'get_bert_score',
    'check_semantic_preservation', 'safe_rewrite', 'cosine_similarity',
    'deep_restructure', 'restructure_sentences', 'bert_naturalness_score',
    'analyze_text', 'compute_lr_score', 'compute_perplexity',
    # config
    'load_config', 'get_config', 'reload_config', 'save_user_config', 'get_data_dir',
]

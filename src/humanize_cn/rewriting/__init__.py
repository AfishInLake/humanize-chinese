from .humanize import (
    humanize, remove_three_part_structure, replace_phrases,
    reduce_high_freq_bigrams, inject_noise_expressions,
    randomize_sentence_lengths, merge_short_sentences, split_long_sentences,
    cap_transition_density, diversify_vocabulary, zipf_perturb,
    vary_paragraph_rhythm, reduce_punctuation, add_casual_expressions,
    shorten_paragraphs, pick_best_replacement, expand_with_cilin,
)
from .style import (
    apply_style, list_styles, transform_casual, transform_zhihu,
    transform_xiaohongshu, transform_wechat, transform_academic,
    transform_literary, transform_novel, transform_weibo,
    add_emojis, strip_emojis,
)
from .paragraph import humanize_paragraph as paragraph_humanize

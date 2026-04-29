"""定向改写策略 — 针对 AIGC 检测统计特征的精准打击

三种策略组合：
1. 低频 n-gram 注入：在低 perplexity 片段中扩展表达，直接改变字符级统计分布
2. Perplexity 定向改写：找出最 AI 的句子做结构性改写
3. 反馈闭环：改写→检测→调整循环，确保每次改写都在降低分数

核心思路：检测特征是字符级的（perplexity、surprisal 分布），
改写也必须在字符级操作，而非词级同义词替换。
"""

import re
import random
import math

# Lazy imports to avoid circular dependencies
_compute_perplexity = None
_bigram_freq = None
_load_freq_fn = None


def _ensure_deps():
    """延迟加载依赖，避免循环导入。"""
    global _compute_perplexity, _bigram_freq, _load_freq_fn
    if _compute_perplexity is None:
        try:
            from ..models.ngram import compute_perplexity as _cp, _load_freq as _lf
            _compute_perplexity = _cp
            _load_freq_fn = _lf
            freq = _lf()
            _bigram_freq = freq.get('bigrams', {})
        except ImportError:
            _compute_perplexity = lambda *a, **k: {'perplexity': 0, 'log_probs': []}
            _bigram_freq = {}


def _count_chinese(text):
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')


def _split_sentences(text):
    """按句号、感叹号、问号、换行分割句子。"""
    parts = re.split(r'([。！？\n])', text)
    result = []
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ''
        if len(sent.strip()) > 1:
            result.append(sent + punct)
    if len(parts) % 2 == 1 and parts[-1].strip():
        result.append(parts[-1])
    return result


def _get_sentence_perplexity(sentence):
    """获取单句的 perplexity。"""
    _ensure_deps()
    result = _compute_perplexity(sentence, window_size=0)
    return result.get('perplexity', 0)


def _get_window_perplexities(text, window_size=40):
    """获取文本的窗口级 perplexity 列表。"""
    _ensure_deps()
    result = _compute_perplexity(text, window_size=window_size)
    return result.get('window_perplexities', [])


def _bigram_freq_score(text):
    """计算文本的平均 bigram 频率（越低越好，表示越不可预测）。"""
    _ensure_deps()
    if not _bigram_freq:
        return 0
    chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chars) < 3:
        return 0
    total = 0
    count = 0
    for i in range(len(chars) - 1):
        bg = chars[i] + chars[i + 1]
        total += _bigram_freq.get(bg, 0)
        count += 1
    return total / count if count > 0 else 0


# ═══════════════════════════════════════════════════════════════════
#  策略 1：低频 n-gram 注入
# ═══════════════════════════════════════════════════════════════════

# 安全的短语级替换规则
# 格式：(源短语, [候选替换短语], ...)
# 只替换完整的短语，不会匹配到复合词内部
_SAFE_PHRASE_RULES = [
    # ── 动+宾 扩展 ──
    ('简化操作', ['对操作加以简化', '在操作层面做简化处理']),
    ('简化流程', ['对流程加以简化', '在流程上做精简']),
    ('简化数据库操作', ['对数据库操作加以简化', '在数据库操作方面做简化处理']),
    ('提升效率', ['在效率上有所提升', '促进效率的提升']),
    ('提升性能', ['在性能上有所提升', '促进性能的提升']),
    ('提升用户体验', ['在用户体验方面有所提升', '促进用户体验的提升']),
    ('优化性能', ['对性能进行优化', '在性能层面做进一步优化']),
    ('优化查询', ['对查询进行优化', '在查询层面做进一步优化']),
    ('优化算法', ['对算法进行优化', '在算法层面做进一步优化']),
    ('确保安全', ['从机制上保证安全', '为安全提供保障']),
    ('确保数据安全', ['从机制上保证数据安全', '为数据安全提供保障']),
    ('确保系统稳定', ['从机制上保证系统稳定运行', '为系统稳定运行提供保障']),
    ('支持多条件', ['为多条件提供支撑', '在多条件方面给予支持']),
    ('支持跨浏览器', ['为跨浏览器访问提供支撑', '在跨浏览器方面给予支持']),
    ('包含注册', ['涵盖了注册', '将注册纳入其中']),
    ('包含用户', ['涵盖了用户', '将用户纳入其中']),
    ('利用技术', ['依托技术来推进', '以技术作为基础']),
    ('利用框架', ['依托框架来推进', '以框架作为基础']),
    ('采用技术', ['选用技术作为技术方案', '以技术为支撑']),
    ('采用框架', ['选用框架作为技术方案', '以框架为支撑']),
    ('设计接口', ['对接口做了专门设计', '在接口的设计上有所考量']),
    ('设计规范', ['对规范做了专门设计', '在规范的设计上有所考量']),
    ('管理用户', ['对用户进行统一管理', '在用户的管理上建立规范']),
    ('管理订单', ['对订单进行统一管理', '在订单的管理上建立规范']),
    ('管理商品', ['对商品进行统一管理', '在商品的管理上建立规范']),
    ('维护信息', ['负责信息的日常维护', '对信息做持续性维护']),
    ('维护数据', ['负责数据的日常维护', '对数据做持续性维护']),
    ('控制访问', ['对访问实施管控', '在访问方面做好把控']),
    ('控制权限', ['对权限实施管控', '在权限方面做好把控']),
    ('处理请求', ['对请求做相应处理', '针对请求给出应对方案']),
    ('处理数据', ['对数据做相应处理', '针对数据给出应对方案']),
    ('存储数据', ['将数据持久化保存', '对数据做持久化存储']),
    ('存储文件', ['将文件持久化保存', '对文件做持久化存储']),
    ('生成订单', ['自动生成订单', '完成订单的创建']),
    ('生成报告', ['自动生成报告', '完成报告的创建']),
    ('展示信息', ['将信息呈现给用户', '在前端呈现信息']),
    ('展示数据', ['将数据呈现给用户', '在前端呈现数据']),
    ('记录日志', ['对日志做留存记录', '将日志写入文件']),
    ('防止攻击', ['从技术层面规避攻击', '对攻击做好防范']),
    ('防止注入', ['从技术层面规避注入', '对注入做好防范']),
    ('验证用户', ['对用户进行校验', '就用户做核实']),
    ('验证表单', ['对表单进行校验', '就表单做核实']),
    ('验证输入', ['对输入进行校验', '就输入做核实']),

    # ── 形+名 扩展 ──
    ('核心模块', ['最为关键的模块', '起关键作用的模块']),
    ('核心功能', ['最为关键的功能', '起关键作用的功能']),
    ('核心算法', ['最为关键的算法', '起关键作用的算法']),
    ('完整系统', ['较为完备的系统', '体系化的系统']),
    ('完整平台', ['较为完备的平台', '体系化的平台']),
    ('安全机制', ['在安全性上有保障的机制', '可靠且安全的机制']),
    ('安全防护', ['在安全性上有保障的防护', '可靠且安全的防护']),
    ('高效运行', ['在效率上有优势的运行', '运行效率较高的状态']),
    ('高效处理', ['在效率上有优势的处理', '运行效率较高的处理']),
    ('详细描述', ['较为细致的描述', '内容充实的描述']),
    ('详细分析', ['较为细致的分析', '内容充实的分析']),
    ('有效保障', ['切实可行的保障', '能产生实效的保障']),
    ('有效提升', ['切实可行的提升', '能产生实效的提升']),

    # ── 连接词扩展 ──
    ('，并', ['，与此同时，还']),
    ('，同时', ['，在此过程中，']),
    ('，此外', ['，在此基础上，']),
    ('，另外', ['，在另一层面上，']),
    ('，从而', ['，以此方式来']),
    ('，因此', ['，基于上述原因，']),
    ('通过技术', ['借助于技术']),
    ('通过框架', ['借助于框架']),
    ('通过算法', ['借助于算法']),
    ('基于需求', ['立足于需求']),
    ('基于数据', ['立足于数据']),
    ('基于规则', ['立足于规则']),
    ('围绕核心', ['以核心为中心']),
    ('围绕用户', ['以用户为中心']),
]


def _find_phrase_match(text, phrase):
    """在文本中查找短语匹配，确保是完整的词匹配（不是子串）。

    检查匹配位置前后是否是汉字，如果是则跳过（避免匹配复合词内部）。
    """
    idx = 0
    while True:
        pos = text.find(phrase, idx)
        if pos == -1:
            return None
        # 检查前面：应该是文本开头或非汉字
        before_ok = (pos == 0 or not ('\u4e00' <= text[pos - 1] <= '\u9fff'))
        # 检查后面：应该是文本结尾或非汉字
        after_pos = pos + len(phrase)
        after_ok = (after_pos >= len(text) or not ('\u4e00' <= text[after_pos] <= '\u9fff'))
        if before_ok and after_ok:
            return pos
        idx = pos + 1
    return None


def inject_low_freq_ngrams(text, max_injections=5, scene='academic'):
    """在低 perplexity 片段中注入低频 n-gram。

    通过短语级替换（"简化操作" → "对操作加以简化"）来：
    1. 增加文本字数，改变句长分布
    2. 引入新的 bigram 组合，降低平均 bigram 频率
    3. 改变字符级 n-gram 分布，直接影响 perplexity

    Args:
        text: 输入文本
        max_injections: 每段最多注入次数
        scene: 场景（academic/general）
    Returns:
        改写后的文本
    """
    _ensure_deps()
    if _count_chinese(text) < 30:
        return text

    # 1. 找出 perplexity 最低的句子（最 AI 的句子）
    sentences = _split_sentences(text)
    if len(sentences) < 1:
        return text

    sent_ppls = []
    for sent in sentences:
        ppl = _get_sentence_perplexity(sent)
        sent_ppls.append((sent, ppl))

    # 按 perplexity 升序排序（最低的 = 最 AI 的）
    sent_ppls.sort(key=lambda x: x[1])

    # 2. 对低 perplexity 句子应用短语替换规则
    injected = 0
    result_sentences = {id(sent): sent for sent in sentences}

    for sent, ppl in sent_ppls:
        if injected >= max_injections:
            break
        if ppl <= 0:
            continue

        original = result_sentences[id(sent)]
        new_sent = original

        # 尝试每条规则，选择 perplexity 提升最大的
        best_sent = None
        best_ppl_gain = 0

        for phrase, candidates in _SAFE_PHRASE_RULES:
            pos = _find_phrase_match(new_sent, phrase)
            if pos is None:
                continue

            for candidate in candidates:
                # 替换
                candidate_sent = new_sent[:pos] + candidate + new_sent[pos + len(phrase):]

                # 验证：中文字数增加不超过 30%
                orig_cn = _count_chinese(new_sent)
                cand_cn = _count_chinese(candidate_sent)
                if cand_cn > orig_cn * 1.3:
                    continue

                # 验证：perplexity 确实提升
                cand_ppl = _get_sentence_perplexity(candidate_sent)
                ppl_gain = cand_ppl - _get_sentence_perplexity(new_sent)

                if ppl_gain > best_ppl_gain and cand_ppl > 0:
                    best_ppl_gain = ppl_gain
                    best_sent = candidate_sent

        if best_sent is not None and best_ppl_gain > 0.5:
            result_sentences[id(sent)] = best_sent
            injected += 1

    # 3. 重建文本（保持原始句子顺序）
    result = []
    for sent in sentences:
        result.append(result_sentences[id(sent)])
    return ''.join(result)


# ═══════════════════════════════════════════════════════════════════
#  策略 2：Perplexity 定向结构性改写
# ═══════════════════════════════════════════════════════════════════

# 结构性改写规则 — 短语级匹配，避免复合词问题
_STRUCTURAL_PHRASE_RULES = [
    # ── 拆分规则：在连接词处拆分长句 ──
    ('，同时', '。同时'),
    ('，并且', '。并且'),
    ('，此外，', '。此外，'),
    ('，另外，', '。另外，'),
    ('，在此基础上，', '。在此基础上，'),
    ('，在此基础上还', '。在此基础上还'),

    # ── 表达转换 ──
    ('的优势，', '所具备的优势，'),
    ('的优势。', '所具备的优势。'),
    ('的特点，', '呈现出的特点，'),
    ('的特点。', '呈现出的特点。'),
    ('的作用，', '所发挥的作用，'),
    ('的作用。', '所发挥的作用。'),
    ('的目的，', '所要达成的目的，'),
    ('的目的。', '所要达成的目的。'),
    ('的需求，', '所对应的需求，'),
    ('的需求。', '所对应的需求。'),
    ('的功能，', '所承担的功能，'),
    ('的功能。', '所承担的功能。'),
    ('的效果，', '所产生的效果，'),
    ('的效果。', '所产生的效果。'),
    ('的结果，', '所得到的结果，'),
    ('的结果。', '所得到的结果。'),
]


def structural_rewrite_sentence(sentence):
    """对单个句子做结构性改写，返回改写后的版本。

    尝试所有结构规则，返回 perplexity 提升最大的版本。
    如果没有提升，返回原文。
    """
    _ensure_deps()

    original_ppl = _get_sentence_perplexity(sentence)
    if original_ppl <= 0:
        return sentence

    best = sentence
    best_ppl = original_ppl

    for phrase, replacement in _STRUCTURAL_PHRASE_RULES:
        pos = _find_phrase_match(sentence, phrase)
        if pos is None:
            continue

        candidate = sentence[:pos] + replacement + sentence[pos + len(phrase):]

        # 验证中文字数变化合理
        orig_cn = _count_chinese(sentence)
        cand_cn = _count_chinese(candidate)
        if cand_cn < orig_cn * 0.7 or cand_cn > orig_cn * 1.4:
            continue

        cand_ppl = _get_sentence_perplexity(candidate)
        if cand_ppl > best_ppl:
            best_ppl = cand_ppl
            best = candidate

    return best


def targeted_structural_rewrite(text, max_rewrites=4):
    """Perplexity 定向改写：找出最 AI 的句子做结构性改写。

    1. 逐句计算 perplexity
    2. 选出 perplexity 最低的 N 个句子
    3. 对每个句子尝试多种结构改写
    4. 只保留 perplexity 确实提升的版本

    Args:
        text: 输入文本
        max_rewrites: 最多改写几个句子
    Returns:
        改写后的文本
    """
    _ensure_deps()
    if _count_chinese(text) < 30:
        return text

    sentences = _split_sentences(text)
    if len(sentences) < 1:
        return text

    # 计算每句 perplexity
    sent_ppls = []
    for i, sent in enumerate(sentences):
        ppl = _get_sentence_perplexity(sent)
        sent_ppls.append((i, sent, ppl))

    # 按 perplexity 升序排序（最低的 = 最 AI 的）
    sent_ppls.sort(key=lambda x: x[2])

    rewritten = 0
    for idx, sent, ppl in sent_ppls:
        if rewritten >= max_rewrites:
            break
        if ppl <= 0:
            continue
        if _count_chinese(sent) < 10:
            continue

        new_sent = structural_rewrite_sentence(sent)
        if new_sent != sent:
            sentences[idx] = new_sent
            rewritten += 1

    return ''.join(sentences)


# ═══════════════════════════════════════════════════════════════════
#  策略 3：反馈闭环改写
# ═══════════════════════════════════════════════════════════════════

def feedback_loop_rewrite(text, detect_fn, score_fn, max_rounds=3, target_drop=5):
    """反馈闭环改写：改写→检测→调整循环。

    每轮：
    1. 对文本应用低频 n-gram 注入 + 结构性改写
    2. 检测改写后分数
    3. 如果分数下降，继续下一轮
    4. 如果分数上升或不变，回退到上一轮的版本

    Args:
        text: 输入文本
        detect_fn: 检测函数，签名 (text) -> (issues, metrics)
        score_fn: 评分函数，签名 (issues, metrics) -> int
        max_rounds: 最大迭代轮数
        target_drop: 目标降幅（分）
    Returns:
        (best_text, original_score, best_score, drop)
    """
    _ensure_deps()

    original_issues, original_metrics = detect_fn(text)
    original_score = score_fn(original_issues, original_metrics)

    best_text = text
    best_score = original_score

    for round_num in range(max_rounds):
        if best_score <= 0:
            break
        if original_score - best_score >= target_drop:
            break

        # 应用改写策略
        candidate = inject_low_freq_ngrams(best_text, max_injections=3, scene='academic')
        candidate = targeted_structural_rewrite(candidate, max_rewrites=2)

        if candidate == best_text:
            break  # 没有变化，停止

        # 检测改写后分数
        new_issues, new_metrics = detect_fn(candidate)
        new_score = score_fn(new_issues, new_metrics)

        if new_score < best_score:
            # 分数下降，接受改写
            best_text = candidate
            best_score = new_score
        # 否则回退（保持 best_text 不变）

    drop = original_score - best_score
    return best_text, original_score, best_score, drop

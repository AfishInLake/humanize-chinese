# humanize-chinese 全面改造计划

> 版本：v1.0 | 日期：2026-04-28 | 状态：待实施

---

## 1. 改造总览

### 1.1 改造目标

**核心指标**：自测通过——使用本项目 `detect_cn.py` 检测改写后的文本，AI 评分降低 **30+ 分**（相对原始 AI 文本）。

**子目标**：
- 词汇替换不改变原意（语义保持检查，BERT ONNX 余弦相似度 >= 0.85）
- 句法变换不产生语法错误（保守优先原则，出错即回退）
- 篇章层噪声插入不影响语义连贯性
- 检测层对短文本（< 100 字）有合理评分能力

### 1.2 改造范围

| 层级 | 模块 | 当前状态 | 改造方向 |
|------|------|----------|----------|
| 词汇层 | `humanize_cn.py` | 硬编码白名单，不区分词性 | 术语配置化 + 词性感知 + Zipf 扰动 |
| 句法层 | `restructure_cn.py` | 45 个正则模板，正则硬切长句 | ltp 依存分析辅助拆分 + 智能句式变换 |
| 篇章层 | `humanize_cn.py` + `restructure_cn.py` | 随机噪声插入，模板少 | 语义感知转折 + 丰富自我修正 + 段落首句多样化 |
| 检测层 | `detect_cn.py` | 短文本全零特征，无语义检查 | 短文本规则评分 + BERT ONNX 语义守卫 |

### 1.3 技术栈

| 工具 | 用途 | 版本要求 |
|------|------|----------|
| **jieba** | 分词 + 词性标注 | >= 0.42 |
| **ltp** (pyltp 或 ltp) | 依存句法分析 | pyltp >= 0.2.1 或 ltp >= 4.0 |
| **onnxruntime** | BERT ONNX 推理 | >= 1.16.0 |
| **transformers** | 仅训练脚本使用 | >= 4.30.0 |
| **torch** | 仅训练脚本使用 | >= 2.0 |

### 1.4 文件结构规划

```
scripts/
├── humanize_cn.py              # 改：词汇层 + 篇章层改造
├── restructure_cn.py           # 改：句法层改造
├── detect_cn.py                # 改：短文本评分修复
├── semantic_guard.py           # 新增：BERT ONNX 语义保持检查
├── protected_terms.json        # 新增：用户自定义术语白名单
├── ngram_model.py              # 不变
├── patterns_cn.json            # 不变
├── train_bert_detector.py      # 新增：BERT 训练脚本（另机运行）
├── export_onnx.py              # 新增：ONNX 导出脚本（另机运行）
└── ...
models/
└── bert_semantic/              # 新增：ONNX 模型目录
    ├── model.onnx              # FP32 或 FP16 模型
    └── tokenizer.json          # tokenizer 文件
```

---

## 2. 词汇层改造

### 2.1 术语白名单保护

#### 当前问题

`humanize_cn.py` 第 298-302 行定义的 `ACADEMIC_PRESERVE_WORDS` 是一个硬编码的 Python `set`：

```python
ACADEMIC_PRESERVE_WORDS = {
    '研究', '分析', '发现', '指出', '表明', '认为', '显示', '揭示',
    '系统', '方法', '结果', '数据', '效果', '作用', '问题', '目标',
    '应用', '提高', '能力', '影响', '过程', '条件',
}
```

**缺陷**：
1. 用户无法自定义——医学论文的"细胞""基因"、法律文本的"合同""条款"等无法保护
2. 仅在 `scene='academic'` 时生效（第 470 行），general/social 场景无保护
3. 修改需要改源码，不利于非技术用户使用

#### 改造方案

**新增 `protected_terms.json` 配置文件**：

```json
{
  "global": ["系统", "方法", "结果", "数据", "效果", "作用", "问题", "目标"],
  "academic": ["研究", "分析", "发现", "指出", "表明", "认为", "显示", "揭示",
               "应用", "提高", "能力", "影响", "过程", "条件"],
  "tech": ["API", "HTTP", "TCP", "JSON", "XML", "算法", "模型", "参数"],
  "legal": ["合同", "条款", "当事人", "义务", "权利", "法律"],
  "medical": ["细胞", "基因", "蛋白质", "临床", "症状", "诊断"],
  "user_custom": []
}
```

**改动文件**：`humanize_cn.py`

**具体代码改动点**：

1. **第 298-302 行附近**——替换硬编码为配置加载：

```python
# 原代码（删除）：
ACADEMIC_PRESERVE_WORDS = {
    '研究', '分析', '发现', ...
}

# 新代码：
_PROTECTED_TERMS_FILE = os.path.join(SCRIPT_DIR, 'protected_terms.json')

def _load_protected_terms():
    """加载术语保护白名单。优先从 JSON 文件加载，回退到内置默认值。"""
    defaults = {
        'global': {'系统', '方法', '结果', '数据', '效果', '作用', '问题', '目标'},
        'academic': {'研究', '分析', '发现', '指出', '表明', '认为', '显示', '揭示',
                     '应用', '提高', '能力', '影响', '过程', '条件'},
    }
    if os.path.exists(_PROTECTED_TERMS_FILE):
        try:
            with open(_PROTECTED_TERMS_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # 合并：JSON 中的值覆盖默认值
            for key in defaults:
                if key in cfg:
                    defaults[key] = set(cfg[key])
            # user_custom 追加到 global
            if 'user_custom' in cfg:
                defaults['global'].update(cfg['user_custom'])
        except (json.JSONDecodeError, OSError):
            pass
    return defaults

_PROTECTED_TERMS = _load_protected_terms()
```

2. **第 470 行**——`reduce_high_freq_bigrams()` 中构建 `preserve` 集合：

```python
# 原代码：
preserve = ACADEMIC_PRESERVE_WORDS if scene == 'academic' else set()

# 新代码：
preserve = _PROTECTED_TERMS.get(scene, set()) | _PROTECTED_TERMS['global']
```

3. **第 582 行**——`_simple_synonym_pass()` 中同步修改：

```python
# 原代码：
preserve = ACADEMIC_PRESERVE_WORDS if scene == 'academic' else set()

# 新代码：
preserve = _PROTECTED_TERMS.get(scene, set()) | _PROTECTED_TERMS['global']
```

4. **第 1216-1232 行**——`diversify_vocabulary()` 中跳过保护词：

```python
# 在 diversity_map 循环前增加：
for word in list(diversity_map.keys()):
    if word in _PROTECTED_TERMS['global']:
        del diversity_map[word]
```

---

### 2.2 词性感知替换

#### 当前问题

`WORD_SYNONYMS`（第 154-273 行）不区分词性。例如"应用"可以是动词（应用技术）或名词（应用场景），当前替换逻辑不检查上下文词性，导致：

- "应用身份证件" → "施用身份证件"（"施用"是动词，"身份证件"是名词宾语，语法不通）
- "有效证件" 已被移除（第 227-229 行注释说明了此问题），但根本原因——缺少词性检查——未解决

`pick_best_replacement()`（第 116-139 行）仅用困惑度排名，不考虑词性匹配。

#### 改造方案

**用 jieba 词性标注，替换前检查候选词与原词词性是否匹配。**

**jieba 词性标注集**（精简版）：

| jieba POS 标签 | 含义 | 示例 |
|----------------|------|------|
| `n` | 普通名词 | 问题、方法、系统 |
| `v` | 普通动词 | 提高、发展、实现 |
| `a` | 形容词 | 重要、显著、主要 |
| `d` | 副词 | 非常、已经、完全 |
| `c` | 连词 | 然而、虽然、因此 |
| `p` | 介词 | 通过、根据、对于 |
| `u` | 助词 | 的、了、着 |
| `vn` | 名动词 | 影响、作用、应用 |
| `ad` | 副形词 | 进一步 |

**简化 POS 映射规则**：

```python
# jieba POS → 简化大类
_POS_CATEGORY = {
    'n': 'noun', 'nr': 'noun', 'ns': 'noun', 'nt': 'noun', 'nz': 'noun', 'ng': 'noun',
    'vn': 'noun_verb',  # 名动词，可作名词也可作动词
    'v': 'verb', 'vd': 'verb', 'vi': 'verb', 'vx': 'verb',
    'a': 'adj', 'ad': 'adj', 'an': 'adj',
    'd': 'adv',
    'c': 'conj',
    'p': 'prep',
}

# 兼容性矩阵：原词 POS 类别 → 候选词允许的 POS 类别
_POS_COMPATIBLE = {
    'noun': {'noun', 'noun_verb'},
    'noun_verb': {'noun', 'noun_verb', 'verb'},
    'verb': {'verb', 'noun_verb'},
    'adj': {'adj'},
    'adv': {'adv'},
    'conj': {'conj'},
    'prep': {'prep'},
}
```

**改动文件**：`humanize_cn.py`

**具体代码改动点**：

1. **新增词性标注辅助函数**（插入在第 115 行之前）：

```python
import jieba
import jieba.posseg as pseg

def _get_word_pos(word, context_sentence):
    """获取 word 在 context_sentence 中的 jieba 词性标注。

    Returns:
        简化 POS 类别字符串，如 'noun', 'verb', 'adj' 等。
        如果 word 未被分词识别，返回 None。
    """
    words = pseg.cut(context_sentence)
    for w, flag in words:
        if w == word:
            return _POS_CATEGORY.get(flag)
    return None

def _get_candidate_pos(candidate):
    """获取候选词的默认词性（独立标注）。

    用于候选词不在原文中时，通过独立分词获取其词性。
    """
    words = pseg.cut(candidate)
    for w, flag in words:
        if w == candidate:
            return _POS_CATEGORY.get(flag)
    return None

def _pos_compatible(orig_pos, cand_pos):
    """判断原词和候选词的词性是否兼容。"""
    if orig_pos is None or cand_pos is None:
        return True  # 无法判断时放行（保守策略）
    allowed = _POS_COMPATIBLE.get(orig_pos, {orig_pos})
    return cand_pos in allowed
```

2. **修改 `pick_best_replacement()`**（第 116-139 行）——增加词性过滤：

```python
def pick_best_replacement(sentence, old, candidates):
    """从多个候选替换中挑选，增加词性兼容性检查。"""
    if not candidates:
        return ''

    # 词性过滤
    orig_pos = _get_word_pos(old, sentence)
    if orig_pos is not None:
        pos_filtered = [
            c for c in candidates
            if _pos_compatible(orig_pos, _get_candidate_pos(c))
        ]
        if pos_filtered:
            candidates = pos_filtered
        # 如果全部被过滤，保留原候选列表（保守策略）

    if not _USE_STATS or len(candidates) <= 1:
        return random.choice(candidates) if candidates else ''

    # ... 后续困惑度排名逻辑不变 ...
```

3. **修改 `reduce_high_freq_bigrams()`**（第 498-529 行）——在候选排名后增加词性检查：

```python
# 在第 519 行 ranked.sort() 之后，第 520 行 "if not ranked: continue" 之前插入：

# 词性过滤：移除与原词词性不兼容的候选
orig_pos = _get_word_pos(word, text[:200])  # 取前 200 字作为上下文
if orig_pos is not None:
    pos_filtered = [
        (c, f) for c, f in ranked
        if _pos_compatible(orig_pos, _get_candidate_pos(c))
    ]
    if pos_filtered:
        ranked = pos_filtered
```

4. **修改 `_simple_synonym_pass()`**（第 599-606 行）——同样增加词性过滤：

```python
# 在第 602 行 candidates = _filter_candidates_for_scene(...) 之后插入：
orig_pos = _get_word_pos(word, text[max(0,pos-50):pos+len(word)+50])
if orig_pos is not None:
    pos_filtered = [
        c for c in candidates
        if _pos_compatible(orig_pos, _get_candidate_pos(c))
    ]
    if pos_filtered:
        candidates = pos_filtered
```

**性能优化**：jieba 分词结果可缓存。对同一 `sentence`，只分词一次：

```python
_pos_cache = {}

def _get_word_pos_cached(word, context_sentence):
    key = (word, context_sentence)
    if key not in _pos_cache:
        _pos_cache[key] = _get_word_pos(word, context_sentence)
    return _pos_cache[key]
```

---

### 2.3 Zipf 词频扰动

#### 当前问题

`diversify_vocabulary()`（第 1198-1232 行）的设计目标是"减少重复词"，即把出现 > 2 次的词替换为同义词。但这反而让词频分布更均匀——更像 AI 生成文本。人类写作的词频遵循 Zipf 定律（少数高频词 + 大量低频词），故意让某些非核心词重复出现反而更自然。

#### 改造方案

**新增 `zipf_perturb()` 函数**，在 `humanize()` 管线中 `diversify_vocabulary()` 之后调用，对词频分布做逆向扰动。

**具体算法**：

```
1. 用 jieba 分词，统计全文词频
2. 计算每个词的 Zipf 期望频率：f(rank) = C / rank^alpha（alpha ≈ 1.0-1.2）
3. 找出"过于均匀"的词——实际频率接近期望但不够极端的词
4. 选 3-5 个非核心词（不在保护白名单中），提升其频率到 1.5-2x
5. 通过在合适位置插入包含该词的短语来实现频率提升
```

**改动文件**：`humanize_cn.py`

**具体代码改动点**：

1. **新增 `zipf_perturb()` 函数**（插入在 `diversify_vocabulary()` 之后，约第 1233 行）：

```python
def zipf_perturb(text, scene='general'):
    """Zipf 词频扰动：让某些非核心词故意重复，模拟人类 Zipf 分布。

    人类写作的词频分布比 AI 更"极端"——少数词出现频率很高。
    此函数在 diversify_vocabulary() 之后运行，补偿其过度均匀化的倾向。
    """
    import jieba

    # 保护词集合
    preserve = _PROTECTED_TERMS.get(scene, set()) | _PROTECTED_TERMS['global']

    # 分词统计词频
    words = [w for w in jieba.cut(text) if w.strip() and len(w) >= 2]
    if len(words) < 20:
        return text

    freq = {}
    for w in words:
        if '\u4e00' not in w:
            continue
        freq[w] = freq.get(w, 0) + 1

    if len(freq) < 5:
        return text

    # 排序，找"中等频率"的词（出现 2-4 次，不在保护名单）
    candidates = [
        (w, c) for w, c in freq.items()
        if 2 <= c <= 4 and w not in preserve and w not in _AI_PATTERN_BLACKLIST
    ]
    if not candidates:
        return text

    # 选 3-5 个词
    n_boost = min(5, max(3, len(candidates) // 4))
    boost_words = random.sample(candidates, min(n_boost, len(candidates)))

    # 为每个 boost 词找一个自然的插入位置（在已有该词的句子附近）
    # 策略：在段落末尾添加一个包含该词的短评论句
    _BOOST_TEMPLATES = {
        'v': ['说到{w}，这确实值得关注。', '关于{w}，前面已经提到了。'],
        'n': ['{w}这个方面，前面也涉及了。', '回到{w}的话题。'],
        'default': ['再说{w}这件事。', '{w}这一点很重要。'],
    }

    for word, count in boost_words:
        # 获取词性
        pos = _get_candidate_pos(word)
        templates = _BOOST_TEMPLATES.get(pos, _BOOST_TEMPLATES['default'])
        template = random.choice(templates)
        insertion = template.format(w=word)

        # 在文本中找一个合适的位置插入（段落末尾）
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 2:
            # 在中间段落末尾插入
            idx = random.randint(1, len(paragraphs) - 2)
            paragraphs[idx] = paragraphs[idx].rstrip() + insertion
            text = '\n\n'.join(paragraphs)

    return text
```

2. **在 `humanize()` 管线中插入调用**（第 1334 行之后）：

```python
# 原代码（第 1334 行）：
text = diversify_vocabulary(text)

# 新代码：
text = diversify_vocabulary(text)
text = zipf_perturb(text, scene=scene)  # Zipf 扰动，补偿过度均匀化
```

---

## 3. 句法层改造

### 3.1 依存句法拆分

#### 当前问题

`restructure_cn.py` 中的 `split_long_sentences()`（第 491-546 行）使用正则在连接词处硬切：

```python
# 第 524 行：仅匹配 2-10 字的主语 + 不仅...还/也
m = re.search(r'(?P<before>[\u4e00-\u9fff]{2,10})不仅(?P<A>...)[，,]\s*(?:还|也|更)(?P<B>.+)', segment)

# 第 533 行：在"，同时/并且/而且"处拆分
m = re.search(r'(?P<before>.+?)[，,]\s*(?:同时|并且|而且)(?P<after>.+)', segment)

# 第 539 行：在"，从而/进而"处拆分
m = re.search(r'(?P<before>.+?)[，,]\s*(?:从而|进而)(?P<after>.+)', segment)
```

**缺陷**：
1. 只能识别 3 种连接模式，覆盖面窄
2. `(?P<before>.+?)` 是贪婪/非贪婪的折中，可能切错位置
3. 无法识别并列结构（COO 关系），如"他喜欢读书、看电影和听音乐"中的并列分句

#### 改造方案

**用 ltp 依存分析找 COO（并列）关系断点，自然拆分。**

**改动文件**：`restructure_cn.py`

**具体代码改动点**：

1. **新增 ltp 依存分析辅助模块**（插入在文件开头 import 区域之后）：

```python
# ─── LTP 依存句法分析（可选依赖）───
_ltp_parser = None
_LTP_AVAILABLE = False

def _init_ltp():
    """延迟初始化 LTP 依存分析器。"""
    global _ltp_parser, _LTP_AVAILABLE
    if _ltp_parser is not None:
        return _LTP_AVAILABLE
    try:
        from ltp import LTP
        _ltp_parser = LTP(path='small')  # 使用 small 模型（~50MB）
        _LTP_AVAILABLE = True
    except ImportError:
        try:
            from pyltp import Parser, Postagger, Segmentor
            # pyltp 需要手动下载模型
            import os
            ltp_data_dir = os.environ.get('LTP_DATA_DIR', '')
            if ltp_data_dir and os.path.exists(ltp_data_dir):
                segmentor = Segmentor()
                segmentor.load(os.path.join(ltp_data_dir, 'cws.model'))
                postagger = Postagger()
                postagger.load(os.path.join(ltp_data_dir, 'pos.model'))
                parser = Parser()
                parser.load(os.path.join(ltp_data_dir, 'parser.model'))
                _ltp_parser = {
                    'segmentor': segmentor,
                    'postagger': postagger,
                    'parser': parser,
                    'type': 'pyltp',
                }
                _LTP_AVAILABLE = True
            else:
                _LTP_AVAILABLE = False
        except ImportError:
            _LTP_AVAILABLE = False
    return _LTP_AVAILABLE


def _find_coo_split_points(sentence):
    """用 LTP 依存分析找并列（COO）关系断点。

    Returns:
        list of (char_offset, connecting_word) 表示可拆分的位置。
        如果 LTP 不可用，返回空列表。
    """
    if not _init_ltp():
        return []

    try:
        if isinstance(_ltp_parser, dict) and _ltp_parser.get('type') == 'pyltp':
            # pyltp 接口
            seg = _ltp_parser['segmentor']
            pos = _ltp_parser['postagger']
            par = _ltp_parser['parser']
            words = list(seg.segment(sentence))
            postags = list(pos.postag(words))
            arcs = list(par.parse(words, postags))
        else:
            # ltp 4.x 接口
            result = _ltp_parser.dep([sentence])
            words = result[0][0]  # 分词结果
            postags = result[0][1]  # 词性
            arcs = result[0][2]  # 依存弧

        # 找 COO 关系
        split_points = []
        for i, arc in enumerate(arcs):
            head = arc.head if hasattr(arc, 'head') else arc[0]
            relation = arc.relation if hasattr(arc, 'relation') else arc[1]
            if relation == 'COO' and i > 0:
                # COO 关系：第 i 个词和第 head 个词是并列的
                # 在第 i 个词之前插入句号
                offset = sum(len(w) for w in words[:i])
                split_points.append((offset, words[i]))
        return split_points
    except Exception:
        return []
```

2. **修改 `split_long_sentences()`**（第 491-546 行）——增加 ltp 优先路径：

```python
def split_long_sentences(text):
    """在特定连接词处拆分长句为两个短句。

    优先使用 LTP 依存分析找 COO（并列）关系断点。
    如果 LTP 不可用，回退到正则方案。
    """
    parts = re.split(r'([。！？])', text)
    result = []

    for i in range(len(parts)):
        segment = parts[i]
        if re.fullmatch(r'[。！？]', segment):
            result.append(segment)
            continue

        cn_len = len(re.findall(r'[\u4e00-\u9fff]', segment))
        if cn_len < 25:
            result.append(segment)
            continue

        # ── 新增：LTP 依存分析路径 ──
        coo_points = _find_coo_split_points(segment)
        if coo_points and len(coo_points) >= 1:
            # 在第一个 COO 断点处拆分
            offset, conn_word = coo_points[0]
            left = segment[:offset].rstrip('，, ')
            right = segment[offset:].lstrip('，, ')
            if left and right and len(re.findall(r'[\u4e00-\u9fff]', left)) >= 8:
                result.append(f'{left}。{right}')
                continue

        # ── 原有正则路径（回退）──
        # ... 保持原有代码不变 ...
        m = re.search(r'(?P<before>[\u4e00-\u9fff]{2,10})不仅...', segment)
        # ...
```

**回退机制**：
- LTP 不可用（未安装或模型缺失）→ `_init_ltp()` 返回 `False` → `_find_coo_split_points()` 返回空列表 → 自动回退到正则方案
- LTP 分析出错（异常）→ `try/except` 捕获 → 返回空列表 → 回退到正则方案
- LTP 找不到 COO 关系 → `coo_points` 为空 → 回退到正则方案

---

### 3.2 智能句式变换

#### 当前问题

`restructure_cn.py` 中 `_build_templates()`（第 25-435 行）构建了 45 个正则模板，由 `restructure_sentences()`（第 438-484 行）逐句匹配。

**缺陷**：
1. 45 个模板覆盖有限——新句式（如"以X为Y""在X中Y"变体）无法处理
2. 每个模板都是手写正则，维护成本高
3. 模板之间可能有冲突（同一句子匹配多个模板，但代码只取第一个）

#### 改造方案

**用 ltp 依存分析自动识别"通过X实现Y"类句式结构，程序化重组。保留正则模板作为 fallback。**

**改动文件**：`restructure_cn.py`

**具体代码改动点**：

1. **新增依存驱动的句式变换函数**（插入在 `restructure_sentences()` 之前）：

```python
def _dependency_restructure(sentence):
    """基于依存分析的句式变换。

    识别以下结构并重组：
    1. SBV(主语) + VOB(谓语) + POB(介宾) → 调换语序
    2. ADV(状语) + HED(核心) + COO(并列) → 拆分并列
    3. VOB(动宾) 结构过长 → 提取宾语前置

    Returns:
        (transformed, was_transformed) 元组。
        如果变换失败，返回 (sentence, False)。
    """
    if not _init_ltp():
        return sentence, False

    try:
        if isinstance(_ltp_parser, dict) and _ltp_parser.get('type') == 'pyltp':
            seg = _ltp_parser['segmentor']
            pos = _ltp_parser['postagger']
            par = _ltp_parser['parser']
            words = list(seg.segment(sentence))
            postags = list(pos.postag(words))
            arcs = list(par.parse(words, postags))
        else:
            result = _ltp_parser.dep([sentence])
            words = result[0][0]
            postags = result[0][1]
            arcs = result[0][2]

        # 策略 1：识别"通过 + 名词 + 动词"结构（ADV + HED + VOB）
        # "通过技术实现突破" → "突破的实现靠的是技术"
        for i, arc in enumerate(arcs):
            relation = arc.relation if hasattr(arc, 'relation') else arc[1]
            if relation == 'ADV' and words[i] == '通过':
                # 找到"通过"修饰的核心动词
                head = arc.head if hasattr(arc, 'head') else arc[0]
                if head < len(words):
                    verb = words[head]
                    # 找动词的宾语
                    for j, a2 in enumerate(arcs):
                        r2 = a2.relation if hasattr(a2, 'relation') else a2[1]
                        h2 = a2.head if hasattr(a2, 'head') else a2[0]
                        if r2 == 'VOB' and h2 == head:
                            obj = words[j]
                            # 找"通过"和动词之间的名词（工具词）
                            tool_words = words[i+1:head]
                            tool = ''.join(tool_words)
                            if tool and len(tool) >= 2:
                                new_sent = f'{obj}的{verb}，靠的是{tool}'
                                return new_sent, True

        return sentence, False
    except Exception:
        return sentence, False
```

2. **修改 `restructure_sentences()`**（第 438-484 行）——增加依存分析优先路径：

```python
def restructure_sentences(text, strength=0.6):
    """对文本中的句子进行句式结构变换。

    优先使用 LTP 依存分析进行程序化重组。
    如果 LTP 不可用或未匹配到结构，回退到正则模板。
    """
    parts = re.split(r'([。！？])', text)
    result = []

    for i in range(0, len(parts)):
        segment = parts[i]
        if re.fullmatch(r'[。！？]', segment):
            result.append(segment)
            continue

        transformed = False
        cn_len = len(re.findall(r'[\u4e00-\u9fff]', segment))
        if segment.strip() and cn_len >= 10 and random.random() < strength:
            # ── 新增：LTP 依存分析路径 ──
            new_segment, was_transformed = _dependency_restructure(segment)
            if was_transformed:
                new_cn_len = len(re.findall(r'[\u4e00-\u9fff]', new_segment))
                if (len(new_segment.strip()) >= 4 and
                    abs(new_cn_len - cn_len) < cn_len * 0.5):
                    segment = new_segment
                    transformed = True

            # ── 原有正则模板路径（fallback）──
            if not transformed:
                for pattern, replacements in _SENTENCE_TEMPLATES:
                    m = pattern.search(segment)
                    if m:
                        repl_fn = random.choice(replacements)
                        try:
                            new_segment = segment[:m.start()] + repl_fn(m) + segment[m.end():]
                            new_cn_len = len(re.findall(r'[\u4e00-\u9fff]', new_segment))
                            if (len(new_segment.strip()) >= 4 and
                                abs(new_cn_len - cn_len) < cn_len * 0.5):
                                segment = new_segment
                                transformed = True
                        except Exception:
                            pass
                        break

        result.append(segment)

    return ''.join(result)
```

---

## 4. 篇章层改造

### 4.1 语义感知转折插入

#### 当前问题

`inject_noise_expressions()`（第 748-829 行）以固定概率（`density` 参数，默认 0.15）在句子间随机插入噪声表达。**不考虑上下文语义连贯性**——可能在"然而"后面再插入"不过"，导致转折词堆叠。

第 795-796 行的核心逻辑：

```python
if random.random() > density:
    continue
```

仅基于随机概率决定是否插入，不检查当前句和相邻句是否已有转折/连接词。

#### 改造方案

**只在"当前句和下一句之间没有转折词"时才插入。**

**改动文件**：`humanize_cn.py`

**具体代码改动点**：

1. **新增转折词检测函数**（插入在 `inject_noise_expressions()` 之前）：

```python
_TRANSITION_WORDS = {
    '然而', '不过', '但是', '可是', '只是', '同时', '此外', '另外',
    '因此', '所以', '于是', '而且', '并且', '另外', '其次', '再者',
    '总之', '综上', '可见', '显然', '换言之', '也就是说',
}

def _has_transition(sentence):
    """检查句子中是否已包含转折/连接词。"""
    for tw in _TRANSITION_WORDS:
        if tw in sentence:
            return True
    return False
```

2. **修改 `inject_noise_expressions()`**（第 782-827 行）——增加语义连贯性检查：

```python
# 原代码（第 795-796 行）：
if random.random() > density:
    continue

# 新代码：
if random.random() > density:
    continue

# 语义连贯性检查：如果当前句或下一句已有转折词，跳过
if _has_transition(s_text):
    continue
if i + 1 < len(sentences) and _has_transition(sentences[i + 1][0]):
    continue

# 如果插入的是 transition_casual 类别，额外检查前后句
if cat == 'transition_casual':
    # 不在已有转折词的句子附近再插入转折
    if i > 0 and _has_transition(sentences[i - 1][0]):
        continue
```

---

### 4.2 自我修正模板丰富

#### 当前问题

`NOISE_EXPRESSIONS`（第 415-428 行）中 `self_correction` 类只有 7 个模板：

```python
'self_correction': ['或者说', '准确地讲', '换个角度看', '严格来说',
                    '更确切地说', '往深了讲', '细想一下'],
```

`NOISE_ACADEMIC_EXPRESSIONS`（第 433-437 行）中只有 4 个：

```python
'self_correction': ['准确地讲', '严格来说', '更确切地说', '往深了讲'],
```

模板数量少导致重复率高，容易被检测器识别为固定模式。

#### 改造方案

**扩充到 20+ 个，按上下文类型分类。**

**改动文件**：`humanize_cn.py`

**具体代码改动点**：

1. **替换 `NOISE_EXPRESSIONS` 中的 `self_correction`**（第 418-419 行）：

```python
# 原代码：
'self_correction': ['或者说', '准确地讲', '换个角度看', '严格来说',
                    '更确切地说', '往深了讲', '细想一下'],

# 新代码：按上下文类型分类
'self_correction': [
    # 通用
    '或者说', '准确地讲', '换个角度看', '严格来说',
    '更确切地说', '往深了讲', '细想一下',
    # 论点后
    '至少从目前来看是这样', '退一步说', '换个说法',
    '不排除其他可能', '这么理解也不算错',
    # 数据后
    '数字背后其实还有故事', '光看数据还不够',
    '数据本身也在变化', '这个数字仅供参考',
    # 对比后
    '两边各有道理', '不能一概而论',
    '具体情况具体分析', '要看从哪个角度',
    # 补充
    '补充一点', '再补充一句', '顺便提一句',
    '说句题外话', '回到正题',
],
```

2. **替换 `NOISE_ACADEMIC_EXPRESSIONS` 中的 `self_correction`**（第 435 行）：

```python
# 原代码：
'self_correction': ['准确地讲', '严格来说', '更确切地说', '往深了讲'],

# 新代码：
'self_correction': [
    '准确地讲', '严格来说', '更确切地说', '往深了讲',
    '从学理上看', '在更严格的意义上', '至少在当前语境下',
    '需要补充说明的是', '不排除其他解释的可能',
    '这一判断有待进一步验证', '从方法论角度审视',
    '考虑到样本局限性', '在控制变量后重新审视',
],
```

3. **同步修改 `NOISE_ACADEMIC_EXPRESSIONS` 中的 `hedging`**（第 434 行）——扩充：

```python
# 原代码：
'hedging': ['客观地说', '实事求是地讲', '平心而论', '公正地看'],

# 新代码：
'hedging': [
    '客观地说', '实事求是地讲', '平心而论', '公正地看',
    '从现有证据来看', '在可观察的范围内', '基于已有研究',
    '就目前掌握的信息而言', '在合理假设的前提下',
],
```

---

### 4.3 段落长度扰动增强

#### 当前问题

`vary_paragraph_rhythm()`（`humanize_cn.py` 第 1010-1052 行）只做简单的合并/拆分：

- 合并：相邻短段落（< 0.6 倍平均长度）以 40% 概率合并
- 拆分：长段落（> 1.5 倍平均长度）在中间句子处拆分

**缺陷**：不检查段落首句是否重复——AI 文本常见所有段落都以相同句式开头（如"随着...的发展""在...背景下"），这会被 `detect_cn.py` 的 `repetitive_starters` 指标（第 312-324 行）检测到。

#### 改造方案

**增加"段落首句多样化"——避免所有段落都以相同句式开头。**

**改动文件**：`restructure_cn.py`（新增函数，在 `deep_restructure()` 中调用）

**具体代码改动点**：

1. **新增 `diversify_paragraph_openers()` 函数**：

```python
# 段落首句变换模板
_OPENER_TRANSFORMS = [
    # "随着X的发展" → "X在持续推进"
    (re.compile(r'^随着(.{2,15})的(?:不断)?(?:发展|进步|演进)'),
     lambda m: f'{m.group(1)}在持续推进'),
    # "在X的背景下" → "X这个大环境"
    (re.compile(r'^在(.{2,15})的背景下'),
     lambda m: f'{m.group(1)}这个大环境'),
    # "X是Y的重要/关键" → "说到X"
    (re.compile(r'^([^，,。]{2,10})是(.{2,15})的(?:重要|关键|核心)'),
     lambda m: f'说到{m.group(1)}'),
    # "值得注意的是" → 删除
    (re.compile(r'^值得注意的是[，,]?\s*'),
     lambda m: ''),
    # "不难发现" → "可以看到"
    (re.compile(r'^不难发现[，,]?\s*'),
     lambda m: '可以看到，'),
]

def diversify_paragraph_openers(text):
    """段落首句多样化：检测并变换重复的段落开头句式。

    如果超过 2 个段落以相同的前 4 字开头，对后续段落的开头做变换。
    """
    paragraphs = text.split('\n\n')
    if len(paragraphs) < 3:
        return text

    # 提取每个段落的首句前 4 字
    openers = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            openers.append('')
            continue
        # 取第一个句子的前 4 个中文字
        first_sent = re.split(r'[。！？]', p)[0]
        cn_chars = re.findall(r'[\u4e00-\u9fff]', first_sent)
        opener = ''.join(cn_chars[:4])
        openers.append(opener)

    # 找重复的开头
    from collections import Counter
    opener_counts = Counter(openers)
    repeated = {op for op, cnt in opener_counts.items() if cnt >= 2 and op}

    if not repeated:
        return text

    # 对重复开头的段落（第 2 个及之后）做变换
    result = []
    seen = set()
    for i, (para, opener) in enumerate(zip(paragraphs, openers)):
        if opener in repeated and opener in seen:
            # 尝试变换开头
            transformed = False
            for pattern, repl_fn in _OPENER_TRANSFORMS:
                m = pattern.search(para.lstrip())
                if m:
                    new_opener = repl_fn(m)
                    para = para[:m.start()] + new_opener + para[m.end():]
                    transformed = True
                    break
            if not transformed:
                # 通用变换：在开头插入一个过渡短语
                transitions = ['另外，', '再说，', '还有一点，', '除此之外，']
                para = random.choice(transitions) + para.lstrip()
        if opener:
            seen.add(opener)
        result.append(para)

    return '\n\n'.join(result)
```

2. **在 `deep_restructure()` 中调用**（第 1196 行之后，AI 废话删除之后）：

```python
# 原代码（第 1196 行）：
text = remove_ai_fillers(text, delete_prob=delete_prob)

# 新代码：
text = remove_ai_fillers(text, delete_prob=delete_prob)
text = diversify_paragraph_openers(text)  # 段落首句多样化
```

---

## 5. 检测层修复

### 5.1 短文本专用评分

#### 当前问题

`detect_cn.py` 第 336 行：

```python
if ngram_analyze and char_count >= 100:
    ngram_stats = ngram_analyze(text)
```

当 `char_count < 100` 时，所有统计层指标（perplexity、burstiness、entropy_cv、surprisal 等）全部关闭。同时，`main()` 第 759 行的融合公式：

```python
score = round(0.2 * rule_score + 0.8 * lr_result['score'])
```

LR 特征在短文本上也可能全为 0 或异常值，导致评分不准确。

#### 改造方案

**`char_count < 100` 时用纯规则评分，不融合 LR。**

**改动文件**：`detect_cn.py`

**具体代码改动点**：

1. **修改 `main()` 中的融合逻辑**（第 746-761 行）：

```python
# 原代码：
lr_result = None if args.rule_only else compute_lr_score(text, scene=args.scene)

if args.rule_only or lr_result is None:
    score = rule_score
else:
    metrics['_lr'] = lr_result
    if args.lr:
        score = lr_result['score']
    else:  # default: fused
        score = round(0.2 * rule_score + 0.8 * lr_result['score'])
        metrics['_fused'] = {'rule_stat': rule_score, 'lr': lr_result['score']}

# 新代码：
char_count = count_chinese_chars(text)

if char_count < 100:
    # 短文本：纯规则评分，不融合 LR（LR 特征在短文本上不可靠）
    score = rule_score
    # 短文本补充：检查是否有明显的 AI 套话
    short_ai_markers = ['值得注意的是', '综上所述', '不难发现', '总而言之',
                        '与此同时', '在此基础上', '由此可见', '赋能', '闭环']
    marker_count = sum(text.count(m) for m in short_ai_markers)
    if marker_count >= 2:
        score = min(100, score + 15)  # 短文本中出现 2+ AI 套话，加 15 分
    elif marker_count == 1:
        score = min(100, score + 8)
elif args.rule_only:
    score = rule_score
else:
    lr_result = compute_lr_score(text, scene=args.scene)
    if lr_result is None:
        score = rule_score
    else:
        metrics['_lr'] = lr_result
        if args.lr:
            score = lr_result['score']
        else:  # default: fused
            score = round(0.2 * rule_score + 0.8 * lr_result['score'])
            metrics['_fused'] = {'rule_stat': rule_score, 'lr': lr_result['score']}
```

2. **在 `calculate_score()` 中增加短文本保护**（第 534-540 行附近）：

```python
# 短文本（< 100 字）降低情感密度惩罚阈值
if metrics.get('emotional_density', 1) < 0.1:
    if metrics['char_count'] > 500:
        score = min(100, score + 5)
    elif metrics['char_count'] > 100:
        score = min(100, score + 3)
    # < 100 字不惩罚情感密度——短文本不需要情感表达
```

---

### 5.2 语义保持检查（BERT ONNX）

#### 当前问题

改写管线（`humanize()` 函数，第 1255-1423 行）没有检查改写后是否改变原意。极端情况下，多次替换可能使语义偏离。

#### 改造方案

**改写后计算原文与改写文的 BERT embedding 余弦相似度，< 0.85 则回退。**

**改动文件**：新增 `semantic_guard.py`，在 `humanize()` 管线末尾调用。

**详细方案见第 7 节。**

---

## 6. BERT 模型训练方案（另机训练）

> 以下操作在有 GPU 的训练机器上执行，不在本机运行。

### 6.1 训练数据准备

**数据来源**：

| 数据集 | 规模 | 用途 |
|--------|------|------|
| HC3-Chinese | ~20K 对 (AI + 人类) | 主要训练集 |
| CUDRT | ~5K 对 | 补充学术文本 |
| 自采集（新闻/博客/百科） | ~10K | 增加领域覆盖 |

**数据格式**（JSON Lines）：

```jsonl
{"text": "人工智能技术在医疗领域的应用日益广泛...", "label": 0}
{"text": "最近去医院看病，发现挂号全改成了手机操作...", "label": 1}
{"text": "综上所述，人工智能的发展对医疗行业产生了深远影响...", "label": 0}
{"text": "说实话，AI 看病这事儿我觉得还得再看看...", "label": 1}
```

- `label=0`：AI 生成文本
- `label=1`：人类撰写文本

**数据清洗**：

```python
def clean_text(text):
    """清洗训练文本。"""
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除纯数字行
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    # 过滤过短文本（< 20 字）
    if len(re.findall(r'[\u4e00-\u9fff]', text)) < 20:
        return None
    # 过滤过长文本（> 2000 字），截断
    if len(text) > 2000:
        text = text[:2000]
    return text
```

**数据增强**：
- 回译增强：中文 → 英文 → 中文
- 同义词替换：使用本项目 `WORD_SYNONYMS` 做小比例替换
- 截断增强：从长文本中随机截取 100-500 字的片段

### 6.2 模型选择

**推荐：`bert-base-chinese`**

| 模型 | 参数量 | 词表大小 | 推理速度 (CPU) | 推荐理由 |
|------|--------|----------|----------------|----------|
| `bert-base-chinese` | 110M | 21128 | ~50ms/句 | 中文 NLP 基线，社区支持好 |
| `MacBERT-base-chinese` | 110M | 21128 | ~50ms/句 | MLM 纠错预训练，对噪声更鲁棒 |
| `bert-base-multilingual-cased` | 110M | 119547 | ~60ms/句 | 多语言，中文效果略差 |

**不推荐更大模型的原因**：
- 本机推理使用 CPU，大模型（如 `bert-large-chinese`，340M 参数）推理延迟不可接受
- ONNX 导出后模型体积过大（FP32 下 > 1.3GB）
- 中文 AI 检测任务不需要更强的语义理解能力，110M 参数已足够

**最终选择**：`MacBERT-base-chinese`（hfl/chinese-macbert-base），因为其 MLM 纠错预训练对改写后的文本（可能有小幅语义偏移）更鲁棒。

### 6.3 训练流程

**完整训练脚本**（`train_bert_detector.py`）：

```python
#!/usr/bin/env python3
"""
BERT 中文 AI 文本检测器训练脚本
在有 GPU 的训练机器上运行。

用法：
    python train_bert_detector.py \
        --train data/train.jsonl \
        --dev data/dev.jsonl \
        --output models/bert_detector \
        --epochs 5 \
        --batch_size 32 \
        --max_length 512
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class AITextDataset(Dataset):
    """AI 文本检测数据集。"""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                text = item['text']
                label = item['label']
                if len(text.strip()) < 10:
                    continue
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """计算评估指标。"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
    }


def main():
    parser = argparse.ArgumentParser(description='训练 BERT 中文 AI 文本检测器')
    parser.add_argument('--train', required=True, help='训练集路径 (JSONL)')
    parser.add_argument('--dev', required=True, help='验证集路径 (JSONL)')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--model_name', default='hfl/chinese-macbert-base',
                        help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup 比例')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='早停耐心值')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # 加载数据
    train_dataset = AITextDataset(args.train, tokenizer, args.max_length)
    dev_dataset = AITextDataset(args.dev, tokenizer, args.max_length)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output, 'logs'),
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    # 训练
    trainer.train()

    # 评估
    results = trainer.evaluate()
    print(f"\n最终评估结果:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # 保存
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n模型已保存到 {args.output}")


if __name__ == '__main__':
    main()
```

**超参数配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_length` | 512 | BERT 最大输入长度 |
| `epochs` | 5 | 训练轮数（配合早停） |
| `batch_size` | 32 | GPU 显存允许的情况下尽量大 |
| `learning_rate` | 2e-5 | BERT 微调标准学习率 |
| `warmup_ratio` | 0.1 | 前 10% 步数做学习率 warmup |
| `weight_decay` | 0.01 | L2 正则化 |
| `early_stopping_patience` | 3 | 验证集 F1 连续 3 轮不提升则停止 |
| `gradient_accumulation_steps` | 2 | 等效 batch_size=64 |

**评估指标**：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Accuracy | >= 0.92 | 整体准确率 |
| F1 (weighted) | >= 0.91 | 加权 F1 |
| Precision | >= 0.90 | 精确率 |
| Recall | >= 0.90 | 召回率 |

### 6.4 导出 ONNX

**导出脚本**（`export_onnx.py`）：

```python
#!/usr/bin/env python3
"""
将训练好的 BERT 模型导出为 ONNX 格式。
在有 GPU 的训练机器上运行，然后将 .onnx 文件复制到本机。

用法：
    python export_onnx.py \
        --model_dir models/bert_detector \
        --output models/bert_semantic/model.onnx \
        --quantize fp16
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel


def export_onnx(model_dir, output_path, quantize='none'):
    """导出 BERT 模型为 ONNX 格式。"""
    print(f"加载模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    # 创建 dummy input
    dummy_input = tokenizer(
        "这是一段测试文本。", return_tensors="pt",
        max_length=512, padding='max_length', truncation=True,
    )

    # 导出 ONNX
    print(f"导出 ONNX: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # 量化
    if quantize == 'fp16':
        print("执行 FP16 量化...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        fp16_path = output_path.replace('.onnx', '_fp16.onnx')
        quantize_dynamic(
            output_path, fp16_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"FP16 模型已保存: {fp16_path}")
        output_path = fp16_path
    elif quantize == 'int8':
        print("执行 INT8 量化...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = output_path.replace('.onnx', '_int8.onnx')
        quantize_dynamic(
            output_path, int8_path,
            weight_type=QuantType.QInt8,
        )
        print(f"INT8 模型已保存: {int8_path}")
        output_path = int8_path

    # 保存 tokenizer
    tokenizer_path = os.path.join(os.path.dirname(output_path), 'tokenizer.json')
    tokenizer.save_pretrained(os.path.dirname(output_path))

    # 打印模型大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n模型大小: {size_mb:.1f} MB")
    print(f"模型路径: {output_path}")
    print(f"Tokenizer 路径: {os.path.dirname(output_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出 BERT ONNX 模型')
    parser.add_argument('--model_dir', required=True, help='训练好的模型目录')
    parser.add_argument('--output', required=True, help='输出 ONNX 文件路径')
    parser.add_argument('--quantize', choices=['none', 'fp16', 'int8'],
                        default='none', help='量化选项')
    args = parser.parse_args()
    export_onnx(args.model_dir, args.output, args.quantize)
```

**模型大小对比**：

| 格式 | 大小 | 推理速度 (CPU, 单句) | 精度损失 |
|------|------|----------------------|----------|
| FP32 | ~420 MB | ~50ms | 无 |
| FP16 | ~210 MB | ~35ms | < 0.1% |
| INT8 | ~110 MB | ~25ms | ~0.5% |

**推荐**：使用 **FP16** 量化——体积减半，速度提升 30%，精度损失可忽略。

---

## 7. ONNX 推理方案（本机部署）

### 7.1 依赖安装

```bash
# 安装 onnxruntime（CPU 版本）
pip install onnxruntime>=1.16.0

# 如果需要 GPU 加速（可选）
pip install onnxruntime-gpu>=1.16.0

# 安装 transformers（仅用于 tokenizer）
pip install transformers>=4.30.0

# 安装 jieba（词汇层改造）
pip install jieba>=0.42

# 安装 LTP（句法层改造，二选一）
pip install ltp>=4.0          # 推荐：pip 安装，自带模型
# 或
pip install pyltp>=0.2.1     # 需要 LTP_DATA_DIR 环境变量
```

### 7.2 推理接口

**`semantic_guard.py` 完整实现**：

```python
#!/usr/bin/env python3
"""
语义保持检查模块 — BERT ONNX 推理

在 humanize() 管线末尾调用，检查改写后文本是否保持原意。
如果原文与改写文的 BERT embedding 余弦相似度 < 阈值，则回退到改写前版本。

用法：
    from semantic_guard import SemanticGuard
    guard = SemanticGuard()
    is_safe, similarity = guard.check(original_text, rewritten_text)
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'bert_semantic')
_DEFAULT_MODEL_PATH = os.path.join(_MODEL_DIR, 'model_fp16.onnx')
_FALLBACK_MODEL_PATH = os.path.join(_MODEL_DIR, 'model.onnx')

# 模块级缓存
_session = None
_tokenizer = None
_AVAILABLE = None


def _init_onnx():
    """延迟初始化 ONNX Runtime 会话和 tokenizer。"""
    global _session, _tokenizer, _AVAILABLE

    if _AVAILABLE is not None:
        return _AVAILABLE

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # 查找模型文件
        model_path = None
        for path in [_DEFAULT_MODEL_PATH, _FALLBACK_MODEL_PATH]:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            _AVAILABLE = False
            return False

        # 创建 ONNX 会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # 限制线程数，避免占用过多 CPU
        _session = ort.InferenceSession(model_path, sess_options)

        # 加载 tokenizer
        tokenizer_path = _MODEL_DIR
        if os.path.exists(os.path.join(tokenizer_path, 'tokenizer.json')):
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            _tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')

        _AVAILABLE = True
        return True

    except ImportError:
        _AVAILABLE = False
        return False
    except Exception as e:
        import logging
        logging.warning(f'SemanticGuard 初始化失败: {e}')
        _AVAILABLE = False
        return False


def _get_embedding(text, max_length=128):
    """获取文本的 BERT [CLS] embedding。

    Args:
        text: 输入文本
        max_length: 最大序列长度（语义检查不需要 512，128 足够）

    Returns:
        numpy array of shape (hidden_size,)
    """
    encoded = _tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np',
    )

    inputs = {
        'input_ids': encoded['input_ids'].astype(np.int64),
        'attention_mask': encoded['attention_mask'].astype(np.int64),
    }

    outputs = _session.run(None, inputs)
    # outputs[1] 是 pooler_output，shape (1, hidden_size)
    return outputs[1].squeeze(0)


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticGuard:
    """语义保持检查器。"""

    def __init__(self, threshold=0.85):
        """
        Args:
            threshold: 余弦相似度阈值。低于此值视为语义偏移，需要回退。
        """
        self.threshold = threshold
        self._initialized = _init_onnx()

    @property
    def available(self):
        """ONNX 模型是否可用。"""
        return self._initialized

    def check(self, original, rewritten, max_length=128):
        """检查改写是否保持原意。

        Args:
            original: 原始文本
            rewritten: 改写后文本
            max_length: BERT 最大序列长度

        Returns:
            (is_safe, similarity) 元组。
            is_safe: bool，True 表示语义保持良好
            similarity: float，余弦相似度 [0, 1]
        """
        if not self._initialized:
            return True, 1.0  # 不可用时跳过检查

        try:
            emb_orig = _get_embedding(original, max_length)
            emb_rewrite = _get_embedding(rewritten, max_length)
            sim = cosine_similarity(emb_orig, emb_rewrite)
            return sim >= self.threshold, sim
        except Exception:
            return True, 1.0  # 出错时放行

    def check_batch(self, originals, rewrites, max_length=128):
        """批量语义检查。

        Args:
            originals: 原始文本列表
            rewrites: 改写后文本列表
            max_length: BERT 最大序列长度

        Returns:
            list of (is_safe, similarity) 元组
        """
        if not self._initialized:
            return [(True, 1.0)] * len(originals)

        results = []
        for orig, rew in zip(originals, rewrites):
            results.append(self.check(orig, rew, max_length))
        return results


# 便捷函数
_guard_instance = None


def check_semantic(original, rewritten, threshold=0.85):
    """全局便捷函数：检查改写是否保持原意。

    Args:
        original: 原始文本
        rewritten: 改写后文本
        threshold: 余弦相似度阈值

    Returns:
        (is_safe, similarity) 元组
    """
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = SemanticGuard(threshold=threshold)
    return _guard_instance.check(original, rewritten)
```

**批量推理优化说明**：

当前实现是逐句推理。如果需要批量推理（如 `best_of_n` 模式下多次改写），可以扩展 `check_batch()` 方法，将多个文本 padding 到同一长度后一次性送入 ONNX 推理。对于本项目的使用场景（每次只比较一对文本），逐句推理已足够。

### 7.3 集成到改写管线

**改动文件**：`humanize_cn.py`

**在 `humanize()` 函数中集成语义守卫**（第 1370 行之后，最终清理之前）：

```python
# ── 语义保持检查（BERT ONNX）──
# 在所有改写步骤完成后、最终清理之前检查
if _USE_SEMANTIC_GUARD:
    try:
        from semantic_guard import check_semantic
        is_safe, similarity = check_semantic(original_text, text)
        if not is_safe:
            # 语义偏移过大，回退到改写前版本
            # 但保留安全的改写步骤（phrase replacement + structure cleanup）
            text = _safe_pass_only(original_text, scene=scene)
    except ImportError:
        pass  # semantic_guard.py 不存在时跳过
```

**回退策略**：

1. **ONNX 不可用**（`semantic_guard.py` 不存在或 `onnxruntime` 未安装）→ 跳过检查，正常输出改写结果
2. **ONNX 推理出错**（模型文件损坏、内存不足等）→ `try/except` 捕获异常，跳过检查
3. **语义偏移过大**（相似度 < 0.85）→ 回退到"安全通道"（仅执行短语替换 + 结构清理，不执行噪声插入和激进替换）

**新增 `_safe_pass_only()` 函数**：

```python
def _safe_pass_only(text, scene='general'):
    """安全通道：仅执行不会大幅改变语义的改写步骤。

    用于语义守卫检测到偏移时的回退。
    """
    text = remove_three_part_structure(text)
    text = replace_phrases(text, casualness=0.1)
    text = reduce_punctuation(text)
    return text
```

**新增模块级开关**（第 18 行附近）：

```python
# Module-level flag: whether to use BERT ONNX semantic guard
_USE_SEMANTIC_GUARD = True
```

**在 `main()` 中增加 CLI 参数**（第 1446 行附近）：

```python
parser.add_argument('--no-semantic-guard', action='store_true',
                    help='跳过 BERT 语义保持检查')
```

---

## 8. 实施路线图

### Phase 0：基础设施（预计 1 天）

**目标**：搭建依赖和配置框架

| 任务 | 文件 | 说明 |
|------|------|------|
| 创建 `protected_terms.json` | 新增 | 术语白名单配置文件 |
| 安装 jieba | 环境 | `pip install jieba>=0.42` |
| 安装 ltp | 环境 | `pip install ltp>=4.0` |
| 重构 `ACADEMIC_PRESERVE_WORDS` | `humanize_cn.py` | 改为从 JSON 加载（第 2.1 节） |
| 添加 `--protected-terms` CLI 参数 | `humanize_cn.py` | 允许指定自定义术语文件 |

**验收标准**：
- `protected_terms.json` 可被正确加载
- jieba 分词和词性标注正常工作
- ltp 依存分析可初始化（或优雅降级）

### Phase 1：词汇层（预计 2-3 天）

**目标**：消除词性不匹配和词频过度均匀

| 任务 | 文件 | 说明 |
|------|------|------|
| 实现词性感知替换 | `humanize_cn.py` | 第 2.2 节 |
| 实现 Zipf 词频扰动 | `humanize_cn.py` | 第 2.3 节 |
| 单元测试 | 新增 `test_vocab.py` | 测试词性过滤、Zipf 扰动 |

**验收标准**：
- "应用身份证件"不再被替换为"施用身份证件"
- `diversify_vocabulary()` + `zipf_perturb()` 后词频分布更接近 Zipf
- HC3 benchmark 评分降低 5+ 分

### Phase 2：句法层（预计 3-4 天）

**目标**：增强句式变换覆盖面，自然拆分长句

| 任务 | 文件 | 说明 |
|------|------|------|
| 实现 ltp 依存分析拆分 | `restructure_cn.py` | 第 3.1 节 |
| 实现智能句式变换 | `restructure_cn.py` | 第 3.2 节 |
| 回退机制测试 | 新增 `test_syntax.py` | 测试 ltp 不可用时的回退 |

**验收标准**：
- ltp 可用时使用依存分析，不可用时回退到正则
- 并列结构（COO）能被正确识别和拆分
- 不产生语法错误（人工抽检 100 句）
- HC3 benchmark 评分再降 5+ 分

### Phase 3：篇章层（预计 2 天）

**目标**：提升篇章级连贯性和多样性

| 任务 | 文件 | 说明 |
|------|------|------|
| 语义感知转折插入 | `humanize_cn.py` | 第 4.1 节 |
| 自我修正模板扩充 | `humanize_cn.py` | 第 4.2 节 |
| 段落首句多样化 | `restructure_cn.py` | 第 4.3 节 |

**验收标准**：
- 不再出现转折词堆叠
- `self_correction` 模板 >= 20 个
- 段落首句重复率降低
- HC3 benchmark 评分再降 3+ 分

### Phase 4：检测层（预计 1-2 天）

**目标**：修复短文本评分，增加语义守卫框架

| 任务 | 文件 | 说明 |
|------|------|------|
| 短文本专用评分 | `detect_cn.py` | 第 5.1 节 |
| 创建 `semantic_guard.py` 骨架 | 新增 | 第 7.2 节（不含 ONNX 推理） |

**验收标准**：
- 50 字短文本有合理评分（不再全零）
- `semantic_guard.py` 在 ONNX 不可用时优雅降级

### Phase 5：BERT ONNX（预计 3-5 天，另机训练 2 天）

**目标**：部署语义保持检查

| 任务 | 位置 | 说明 |
|------|------|------|
| 准备训练数据 | 训练机 | 第 6.1 节 |
| 训练 BERT 模型 | 训练机 | 第 6.3 节 |
| 导出 ONNX | 训练机 | 第 6.4 节 |
| 部署 ONNX 模型 | 本机 | 复制模型文件到 `models/bert_semantic/` |
| 集成到 `humanize()` | `humanize_cn.py` | 第 7.3 节 |
| 语义守卫测试 | 新增 `test_semantic.py` | 测试相似度计算和回退逻辑 |

**验收标准**：
- ONNX 推理正常工作（CPU，< 100ms/对）
- 语义偏移时正确回退
- 模型文件 <= 250MB（FP16）

### Phase 6：集成测试（预计 2-3 天）

**目标**：端到端验证，确保改写质量

| 任务 | 说明 |
|------|------|
| HC3-Chinese benchmark | 运行 `run_hc3_benchmark.py`，对比改造前后评分 |
| Long-form benchmark | 运行 `run_longform_benchmark.py` |
| 回归测试 | 确保已有通过的测试不退化 |
| 性能测试 | 单篇 1000 字文本改写延迟 < 5 秒（CPU） |
| 人工抽检 | 随机抽 50 篇改写文本，人工检查语义保持和语法正确性 |

**验收标准**：
- HC3 benchmark 平均评分降低 30+ 分
- 无语法错误（人工抽检通过率 >= 95%）
- 无语义偏移（BERT 相似度 >= 0.85）
- 单篇改写延迟 < 5 秒

---

## 9. 风险与缓解

### 9.1 jieba/ltp 依赖增加启动时间和内存

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| jieba 首次加载词典 ~0.5s | 低 | 可接受，jieba 是轻量级库 |
| ltp 加载模型 ~2-5s，内存 ~200MB | 中 | 延迟初始化（`_init_ltp()`），仅在实际需要时加载 |
| ltp 模型文件 ~50MB 需要下载 | 中 | 首次使用时自动下载，后续缓存到本地 |
| 总内存增加 ~300MB | 中 | 提供 `--no-ltp` CLI 参数，允许用户禁用 ltp |

**缓解代码**：

```python
# humanize_cn.py main() 中增加：
parser.add_argument('--no-ltp', action='store_true',
                    help='禁用 LTP 依存分析（回退到正则方案）')
```

### 9.2 ltp 依存分析可能出错

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| COO 关系识别错误导致错误拆分 | 高 | 拆分后验证：左右两半的中文字数 >= 8，否则放弃拆分 |
| 依存分析对口语/网络用语效果差 | 中 | 仅对 >= 25 中文字的长句使用 ltp，短句用正则 |
| ltp 版本升级导致 API 变化 | 低 | 同时支持 `ltp` 4.x 和 `pyltp` 0.x 两套接口 |

**缓解代码**（已在第 3.1 节中体现）：

```python
# 拆分后验证
if left and right and len(re.findall(r'[\u4e00-\u9fff]', left)) >= 8:
    result.append(f'{left}。{right}')
    continue
# 验证失败 → 回退到正则方案
```

### 9.3 BERT ONNX 模型文件大小

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| FP32 模型 ~420MB，部署包过大 | 中 | 使用 FP16 量化，体积降至 ~210MB |
| ONNX Runtime 依赖 ~30MB | 低 | pip 安装，不影响源码仓库 |
| 模型文件需要单独分发 | 低 | 首次运行时提示下载，或作为可选依赖 |

**缓解策略**：
- 默认不打包模型文件，提供下载脚本
- `semantic_guard.py` 在模型不存在时优雅降级（返回 `True, 1.0`）
- 文档中说明模型下载和放置方法

### 9.4 改写质量回归

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 词性过滤过于严格，导致替换率下降 | 中 | 词性无法判断时放行（`orig_pos is None → True`） |
| Zipf 扰动插入的短句不自然 | 中 | 限制插入位置（段落末尾），模板经过人工审核 |
| 语义守卫回退过于频繁 | 中 | 阈值设为 0.85（较宽松），仅拦截严重偏移 |
| 正则模板 + ltp 变换冲突 | 低 | ltp 优先，正则 fallback；每个句子最多变换一次 |
| `best_of_n` 模式下 N 次改写 + N 次 ONNX 推理延迟过高 | 中 | `best_of_n` 模式下跳过语义守卫（已在 N 次中选最优） |

**回归测试策略**：
- 保留现有 benchmark 脚本（`run_hc3_benchmark.py`、`run_longform_benchmark.py`）
- 每个Phase完成后运行 benchmark，对比改造前后评分
- 如果评分下降（检测分数反而升高），回退该 Phase 的改动

---

## 附录 A：关键函数索引

| 函数名 | 文件 | 行号 | 改造阶段 |
|--------|------|------|----------|
| `pick_best_replacement()` | `humanize_cn.py` | 116-139 | Phase 1（词性过滤） |
| `reduce_high_freq_bigrams()` | `humanize_cn.py` | 453-574 | Phase 1（词性过滤 + 白名单） |
| `_simple_synonym_pass()` | `humanize_cn.py` | 577-609 | Phase 1（词性过滤 + 白名单） |
| `diversify_vocabulary()` | `humanize_cn.py` | 1198-1232 | Phase 1（Zipf 扰动） |
| `inject_noise_expressions()` | `humanize_cn.py` | 748-829 | Phase 3（语义转折） |
| `NOISE_EXPRESSIONS` | `humanize_cn.py` | 415-428 | Phase 3（模板扩充） |
| `NOISE_ACADEMIC_EXPRESSIONS` | `humanize_cn.py` | 433-437 | Phase 3（模板扩充） |
| `humanize()` | `humanize_cn.py` | 1255-1423 | Phase 1/5/7（管线集成） |
| `split_long_sentences()` | `restructure_cn.py` | 491-546 | Phase 2（ltp 拆分） |
| `restructure_sentences()` | `restructure_cn.py` | 438-484 | Phase 2（智能变换） |
| `deep_restructure()` | `restructure_cn.py` | 1163-1218 | Phase 3（首句多样化） |
| `vary_paragraph_rhythm()` | `humanize_cn.py` | 1010-1052 | Phase 3（段落扰动） |
| `detect_patterns()` | `detect_cn.py` | 153-462 | Phase 4（短文本） |
| `calculate_score()` | `detect_cn.py` | 495-540 | Phase 4（短文本） |
| `main()` | `detect_cn.py` | 708-772 | Phase 4（融合公式） |

## 附录 B：新增文件清单

| 文件路径 | 用途 | 阶段 |
|----------|------|------|
| `scripts/protected_terms.json` | 术语白名单配置 | Phase 0 |
| `scripts/semantic_guard.py` | BERT ONNX 语义保持检查 | Phase 4/5 |
| `scripts/train_bert_detector.py` | BERT 训练脚本（另机） | Phase 5 |
| `scripts/export_onnx.py` | ONNX 导出脚本（另机） | Phase 5 |
| `models/bert_semantic/model_fp16.onnx` | BERT ONNX 模型 | Phase 5 |
| `models/bert_semantic/tokenizer.json` | Tokenizer 文件 | Phase 5 |
| `tests/test_vocab.py` | 词汇层单元测试 | Phase 1 |
| `tests/test_syntax.py` | 句法层单元测试 | Phase 2 |
| `tests/test_semantic.py` | 语义守卫单元测试 | Phase 5 |

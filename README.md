# 🔧 中文 AI 文本去痕迹工具

**免费、本地运行、零依赖。检测 + 改写一步到位。**

[![GitHub stars](https://img.shields.io/github/stars/voidborne-d/humanize-chinese?style=flat-square)](https://github.com/voidborne-d/humanize-chinese)
[![ClawHub](https://img.shields.io/badge/clawhub-humanize--chinese-blue?style=flat-square)](https://clawhub.com/skills/humanize-chinese)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6+-blue?style=flat-square)](https://python.org)
[![Claude Code](https://img.shields.io/badge/Claude_Code-compatible-orange?style=flat-square)](#claude-code)

---

## 30 秒看效果

```bash
python scripts/academic_cn.py 论文.txt -o 改后.txt --compare
```

```
原文:    79/100 VERY HIGH
改写后:  12/100 LOW
✅ 降低了 67 分
```

就这么简单。不用注册，不用付费，不用联网。

---

## 改写前后对比

### 学术论文

**改写前** 🔴 79分：
> 本文旨在探讨人工智能对高等教育教学模式的影响，具有重要的理论意义和实践价值。研究表明，人工智能技术已被广泛应用于课堂教学、学生评估和个性化学习等多个方面。

**改写后** 🟢 12分：
> 本研究聚焦于人工智能对高等教育教学模式的影响，兼具理论探索与实践参考的双重价值。前人研究发现，人工智能技术已广泛用于课堂教学、学生评估和个性化学习等多个方面。

### 通用文本

**改写前** 🔴 87分：
> 综上所述，人工智能技术在教育领域具有重要的应用价值和广阔的发展前景。值得注意的是，随着技术的不断发展，AI 将在个性化学习、智能评估等方面发挥越来越重要的作用，为教育行业的数字化转型赋能。

**改写后** 🟢 12分：
> 简单讲，人工智能技术在教育领域有其独特价值和广阔的进展前景。如今，AI 将在个性化学习、智能评估等维度发挥越来越要紧的作用。

### 社交媒体 → 小红书风格

**改写前** 🟠 72分：
> 在当今快节奏的生活中，时间管理具有至关重要的意义。

**改写后** 🟢 8分：
> 姐妹们！说真的，时间管理这事我踩过太多坑了 😭 之前天天加班到半夜，后来摸索出一套方法，现在居然能准点下班了？！

---

## 安装

```bash
# 方式一：ClawHub
clawhub install humanize-chinese

# 方式二：Git Clone
git clone https://github.com/voidborne-d/humanize-chinese.git

# 方式三：Claude Code Skill
npx skills add https://github.com/voidborne-d/humanize-chinese.git
```

不需要 `pip install` 任何东西。下载就能用。

---

## Claude Code

4 个 slash command，复制到 `.claude/commands/` 即可：

```bash
git clone https://github.com/voidborne-d/humanize-chinese.git
cp humanize-chinese/claude-code/*.md YOUR_PROJECT/.claude/commands/
```

然后在 Claude Code 里：

```
/detect 综上所述，人工智能技术在教育领域具有重要的应用价值...
/humanize 本文旨在探讨人工智能对高等教育教学模式的影响...
/academic 论文.txt
/style xiaohongshu 在当今快节奏的生活中...
```

| 命令 | 功能 |
|------|------|
| `/detect` | AI 痕迹检测，0-100 评分 |
| `/humanize` | 去 AI 味改写 |
| `/academic` | 学术论文 AIGC 降重 |
| `/style [风格]` | 风格转换（7 种） |

---

## 快速上手

### 🎓 学术论文降 AIGC 率

```bash
# 检测
python scripts/academic_cn.py 论文.txt

# 改写 + 对比
python scripts/academic_cn.py 论文.txt -o 改后.txt --compare

# 激进模式（降得更狠）
python scripts/academic_cn.py 论文.txt -o 改后.txt -a --compare
```

### 🔍 通用文本去 AI 味

```bash
python scripts/detect_cn.py text.txt -v       # 检测
python scripts/humanize_cn.py text.txt -o clean.txt  # 改写
python scripts/compare_cn.py text.txt -a       # 对比
```

### 🎨 风格转换

```bash
python scripts/style_cn.py text.txt --style xiaohongshu   # 小红书
python scripts/style_cn.py text.txt --style zhihu          # 知乎
python scripts/style_cn.py text.txt --style weibo           # 微博
```

7 种风格：口语化 / 知乎 / 小红书 / 公众号 / 学术 / 文艺 / 微博

---

## 功能一览

| 功能 | 说明 |
|------|------|
| 🔍 AI 检测 | 20+ 规则维度 + N-gram 困惑度统计，0-100 评分，精确到句子 |
| 📈 统计分析 | 困惑度 / 突发度 / 段落熵，从概率分布层面检测 AI |
| ✏️ 智能改写 | 困惑度引导选词 + 低频 bigram 注入 + 句长随机化 + 噪声表达 |
| 🎓 学术降重 | 10 维度检测 + 120 条学术替换，针对知网/维普/万方 |
| 🎨 风格转换 | 7 种中文写作风格一键转换 |
| 📊 前后对比 | 改写前后评分变化一目了然 |
| 🔄 可复现 | `--seed` 保证相同输入相同输出 |
| 📦 零依赖 | 纯 Python 标准库，下载即用 |

---

## 🎓 学生党必看

用 ChatGPT / DeepSeek 写了论文初稿？三步搞定：

```bash
# 1. 看看 AIGC 率多高
python scripts/academic_cn.py 论文.txt

# 2. 一键改写
python scripts/academic_cn.py 论文.txt -o 改后.txt --compare

# 3. 不够就开激进模式
python scripts/academic_cn.py 论文.txt -o 改后.txt -a --compare
```

**工具做了什么：**
- "本文旨在" → "本研究聚焦于"
- "被广泛应用" → "得到较多运用"
- 打破每段一样长的结构
- 加入"可能""在一定程度上"等学术犹豫语
- "研究表明" → "笔者认为""前人研究发现"

⚠️ 改完通读一遍，确认专业术语没被误改、引用格式正确。建议用知网 AMLC 或维普验证。

---

## 评分标准

| 分数 | 等级 | 含义 |
|------|------|------|
| 0-24 | 🟢 LOW | 基本像人写的 |
| 25-49 | 🟡 MEDIUM | 有些 AI 痕迹 |
| 50-74 | 🟠 HIGH | 大概率 AI 生成 |
| 75-100 | 🔴 VERY HIGH | 几乎确定是 AI |

---

## 技术原理

### 两层检测

**规则层**（看词）：三段式套路、机械连接词、空洞宏大词、AI 高频词、模板句式、段落结构均匀度……

**统计层**（看分布）：基于 15 万条中文字符 n-gram 频率表：

| 指标 | AI 文本 | 人类文本 |
|------|---------|----------|
| 困惑度 | ~231（可预测） | ~533（自然） |
| 突发度 | 均匀 | 忽简忽繁 |
| 段落熵 | 每段一样 | 有差异 |

### 智能改写

不是盲目替换。每次替换从多候选中**选困惑度最高的**（最不可预测 = 最像人写的）。

三个深度策略：
1. **低频 bigram 注入**：扫描最高频的字组合，换成低频同义词
2. **句子长度随机化**：制造不均匀节奏（AI 每句差不多长，人类忽长忽短）
3. **噪声表达插入**：犹豫语、自我修正、不确定标记（学术模式自动过滤口语）

效果：

| 文本类型 | 原文 | 改写后 | 激进模式 | 人类参考 |
|---|---|---|---|---|
| 通用 | 231 | 306 | 348 | ~533 |
| 学术 | 174 | 273 | 326 | ~533 |

---

## CLI 参数速查

```bash
# 检测
python scripts/detect_cn.py [file] [-v] [-s] [-j]

# 改写
python scripts/humanize_cn.py [file] [-o out] [--scene S] [--style S] [-a] [--seed N]

# 学术降重
python scripts/academic_cn.py [file] [-o out] [--detect-only] [-a] [--compare]

# 风格转换
python scripts/style_cn.py [file] --style S [-o out]

# 对比
python scripts/compare_cn.py [file] [-o out] [--scene S] [-a]
```

| 参数 | 说明 |
|------|------|
| `-v` | 详细模式，显示最可疑的句子 |
| `-s` | 只输出评分 |
| `-j` | JSON 输出 |
| `-o` | 输出文件 |
| `-a` | 激进模式 |
| `--seed N` | 固定随机种子 |
| `--no-stats` | 关闭统计优化（更快） |
| `--no-noise` | 关闭噪声注入和句长随机化 |

---

## 批量处理

```bash
for f in *.txt; do echo "=== $f ===" && python scripts/detect_cn.py "$f" -s; done
for f in *.md; do python scripts/humanize_cn.py "$f" -a -o "${f%.md}_clean.md"; done
```

---

## 自定义

所有检测模式、替换词库、权重都在 `scripts/patterns_cn.json`，可以自己改。

---

## 对比 Humanizer-zh

和 [Humanizer-zh](https://github.com/op7418/Humanizer-zh)（5k⭐）的区别：

| | 本项目 | Humanizer-zh |
|---|---|---|
| 运行方式 | ✅ 独立 CLI，终端直接跑 | 纯 prompt，必须在 Claude Code 内用 |
| 依赖 | ✅ 零依赖 | 需要 Claude Code + API 额度 |
| 量化评分 | ✅ 0-100 分 | ❌ 无评分 |
| 统计检测 | ✅ N-gram 困惑度 + 突发度 + 熵 | ❌ 无 |
| 学术模式 | ✅ 10 维度 + 120 条替换 | ❌ 无 |
| 风格转换 | ✅ 7 种 | ❌ 无 |
| 可复现 | ✅ `--seed` | ❌ 每次不同 |
| 批量处理 | ✅ CLI 管道 | ❌ 只能单篇交互 |
| 免费 | ✅ 完全免费 | ⚠️ 需要 API 额度 |

简单说：Humanizer-zh 是个好 prompt，但只能在 Claude Code 里用。我们是独立工具，任何环境都能跑。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=voidborne-d/humanize-chinese&type=Date)](https://star-history.com/#voidborne-d/humanize-chinese&Date)

---

## License

MIT — 随便用，不用付钱，不用署名。

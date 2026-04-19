# /humanize — 去除中文文本的 AI 痕迹（v3.0）

Rewrite Chinese text to remove AI fingerprints. Detect → Rewrite → Verify.

v3.0 uses **tiered intensity** (conservative/moderate/full based on input AI score), **40 paraphrase templates**, **45 transition-word replacements**, **paragraph-end short-sentence insertion**, and HC3-calibrated indicator suppression.

## Usage

The user provides Chinese text (directly or as a file path). Run the full pipeline: detect, rewrite, compare.

## Steps

1. Save the input text:
   ```bash
   cat > /tmp/humanize_input.txt << 'HUM_EOF'
   [user's text here]
   HUM_EOF
   ```

2. Run compare (detect + rewrite + score comparison in one step):
   ```bash
   $SKILL_DIR/humanize compare /tmp/humanize_input.txt -a -o /tmp/humanize_output.txt
   # or:
   python $SKILL_DIR/scripts/compare_cn.py /tmp/humanize_input.txt -a -o /tmp/humanize_output.txt
   ```

3. Show the user:
   - Original score → Rewritten score (target < 50 for general, < 40 for academic)
   - The rewritten text
   - Key changes made

## Options

- Default mode sufficient for most cases (auto-selects conservative/moderate/full based on input score)
- Use `-a` (aggressive) to force full-pipeline rewriting
- Add `--style xiaohongshu` / `--style zhihu` etc. for platform-specific rewrites
- Add `--quick` to skip statistical optimization (18× speed on 10k-char text)
- Add `--cilin` to enable CiLin 同义词词林 expansion (38,873 words with semantic filter)

## Available Styles

casual, zhihu, xiaohongshu, wechat, academic, literary, weibo

## Target Scores

| Input type | Good v3.0 output score |
|------------|------------------------|
| Stereotyped AI (样板 AI 文) | 40-55 (MEDIUM, from 90+) |
| Natural ChatGPT | 5-15 (LOW, from 15-25) |
| Academic paper | < 40 (MEDIUM, academic-specific); < 35 generic |

## Example

```
/humanize 本文旨在探讨人工智能对高等教育教学模式的影响，具有重要的理论意义和实践价值。
```

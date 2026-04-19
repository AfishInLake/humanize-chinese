# /detect — 检测中文文本的 AI 痕迹（v3.0）

Detect AI-generated patterns in Chinese text. Score 0-100, 20+ rule dimensions plus 8 HC3-calibrated statistical features (sentence-length CV, short-sentence fraction, comma density, perplexity, GLTR rank buckets, DivEye skew/kurt).

## Usage

The user provides Chinese text (directly or as a file path). Run detection and report results.

## Steps

1. If the user provided a file path, use it directly. Otherwise, save the text to a temp file first:
   ```bash
   cat > /tmp/detect_input.txt << 'DETECT_EOF'
   [user's text here]
   DETECT_EOF
   ```

2. Run detection with verbose mode (use unified CLI if available, else script):
   ```bash
   $SKILL_DIR/humanize detect /tmp/detect_input.txt -v
   # or equivalently:
   python $SKILL_DIR/scripts/detect_cn.py /tmp/detect_input.txt -v
   ```

3. Report the results clearly:
   - Overall score and level (LOW/MEDIUM/HIGH/VERY HIGH)
   - Top suspicious sentences
   - Key AI patterns found (rule-based + statistical indicators)

## Score Reference

| Score | Level | Meaning |
|-------|-------|---------|
| 0–24  | 🟢 LOW | Reads like human-written |
| 25–49 | 🟡 MEDIUM | Some AI traces |
| 50–74 | 🟠 HIGH | Likely AI-generated |
| 75–100 | 🔴 VERY HIGH | Almost certainly AI |

## Statistical Indicators (v3.0)

v3.0 added strong statistical signals calibrated on HC3-Chinese 300+300 samples:

| Indicator | Cohen's d | Description |
|-----------|-----------|-------------|
| `stat_low_sentence_length_cv` | 1.22 | AI writes formulaic 15-25 char sentences |
| `stat_low_short_sentence_fraction` | 1.21 | Humans write short sentences; AI rarely |
| `stat_low_perplexity` | 0.47 | Low character-level trigram perplexity |
| `stat_high_top10_bucket` | 0.44 | AI picks top-probability characters |
| `stat_low_surprisal_skew` | 0.41 | DivEye feature |
| `stat_low_comma_density` | 0.47 | AI writes longer uninterrupted clauses |

## Example

```
/detect 综上所述，人工智能技术在教育领域具有重要的应用价值和广阔的发展前景。
```

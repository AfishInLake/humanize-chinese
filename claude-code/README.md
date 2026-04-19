# Humanize Chinese for Claude Code (v3.0)

## Installation

```bash
# Clone the repo
git clone https://github.com/voidborne-d/humanize-chinese.git

# Copy slash commands to your project
cp -r humanize-chinese/claude-code/*.md YOUR_PROJECT/.claude/commands/
```

Or install individual commands:

```bash
mkdir -p .claude/commands
cp humanize-chinese/claude-code/detect.md .claude/commands/
cp humanize-chinese/claude-code/humanize.md .claude/commands/
cp humanize-chinese/claude-code/academic.md .claude/commands/
cp humanize-chinese/claude-code/style.md .claude/commands/
```

## Commands

| Command | Description |
|---------|-------------|
| `/detect [text]` | Detect AI patterns, 0-100 score. 20+ rules + 8 HC3-calibrated statistical features |
| `/humanize [text]` | Rewrite (tiered conservative/moderate/full based on input score) |
| `/academic [text]` | Academic paper AIGC reduction. 11 dimensions + dual scoring (CNKI/VIP/Wanfang) |
| `/style [style] [text]` | Transform to style (7 styles). Auto-humanizes first. |

## What's new in v3.0

- **HC3 accuracy 73%** (up from 51% in v2.4) — statistical features calibrated on HC3-Chinese 300+300 human/AI pairs
- **Sentence-length CV** (Cohen's d = 1.22, strongest signal)
- **40 paraphrase templates** (up from 15)
- **122 academic-tone replacements** including transitions (首先→其一/最初, 然而→但是, 因此→故而)
- **CiLin synonym expansion** (38,873 words, semantic filter)
- **Tiered humanize intensity** (conservative/moderate/full auto-select)
- **Unified CLI**: `./humanize {detect,rewrite,academic,style,compare}`
- **`--quick` flag**: 18× speed for 10k-char texts

## Note

Make sure the `scripts/` directory from humanize-chinese is accessible. The commands reference `$SKILL_DIR/scripts/` or `$SKILL_DIR/humanize` (the unified CLI shim) — if using as a standalone Claude Code project, update paths to point to your local copy.

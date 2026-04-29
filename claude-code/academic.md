# /academic — 学术论文 AIGC 降重

Reduce AIGC detection scores for Chinese academic papers. 11 detection dimensions + dual scoring.

## Usage

```
/academic [file path or text]
/academic 论文.txt
```

## Steps

1. If text provided directly, save to temp file:
   ```bash
   cat > /tmp/academic_input.txt << 'EOF'
   [text here]
   EOF
   ```

2. Run academic detection + rewrite:
   ```bash
   cd $SKILL_DIR && PYTHONPATH=src python3 -m humanize_cn.interfaces.cli academic /tmp/academic_input.txt -o /tmp/academic_output.txt --compare
   ```

3. Show results:
   - Original score vs rewritten score
   - Key AI patterns detected
   - What was changed

## Options

| Option | Description |
|--------|-------------|
| `--compare` | Show before/after score comparison |
| `--aggressive` | Aggressive rewrite mode |
| `--quick` | Fast mode (skip statistical optimization) |
| `--detect-only` | Only detect, don't rewrite |

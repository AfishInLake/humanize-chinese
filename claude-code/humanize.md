# /humanize — 改写中文文本去除 AI 痕迹

Rewrite Chinese text to remove AI-generated patterns while preserving meaning.

## Usage

```
/humanize [text or file path]
/humanize --scene academic [text]    # 学术论文模式
/humanize --aggressive [text]       # 激进模式
```

## Steps

1. Save input text to temp file:
   ```bash
   cat > /tmp/humanize_input.txt << 'EOF'
   [user's text here]
   EOF
   ```

2. Run humanize:
   ```bash
   cd $SKILL_DIR && PYTHONPATH=src python3 -m humanize_cn.interfaces.cli rewrite /tmp/humanize_input.txt -o /tmp/humanize_output.txt
   ```

   For academic text, use:
   ```bash
   cd $SKILL_DIR && PYTHONPATH=src python3 -m humanize_cn.interfaces.cli academic /tmp/humanize_input.txt -o /tmp/humanize_output.txt --compare
   ```

3. Show the rewritten text.

## Options

| Option | Description |
|--------|-------------|
| `--scene academic` | 学术论文模式 |
| `--scene social` | 社交场景 |
| `--aggressive` | 激进模式 |
| `--quick` | 快速模式（18× 速度） |
| `--cilin` | 启用 CiLin 同义词扩展 |

#!/usr/bin/env python3
"""
AI 文本语料生成脚本。

从 HC3-Chinese 提取问题，调用 Qwen/DeepSeek/Ollama API 生成多种风格的 AI 回答，
用于构建大规模 AI 文本检测训练数据集。

用法:
    # 使用 Qwen API 生成（默认）
    python scripts/generate_ai_corpus.py --api qwen --api-key YOUR_KEY

    # 使用 DeepSeek API 生成
    python scripts/generate_ai_corpus.py --api deepseek --api-key YOUR_KEY

    # 使用 Ollama 本地模型生成（免费，无需 API Key）
    python scripts/generate_ai_corpus.py --api ollama --host http://192.168.0.188:11434 --model qwen3.5:cloud

    # 指定输出文件和并发数
    python scripts/generate_ai_corpus.py --api ollama --host http://localhost:11434 \
        --model qwen3.5:cloud --output ./training_data/ai_generated_v1.jsonl --concurrency 3

    # 快速测试（只生成 10 条）
    python scripts/generate_ai_corpus.py --api ollama --host http://192.168.0.188:11434 \
        --model qwen3.5:cloud --quick

    # 断点续传（默认行为，跳过已处理的 question）
    python scripts/generate_ai_corpus.py --api ollama --host http://192.168.0.188:11434 \
        --model qwen3.5:cloud --resume

依赖:
    pip install httpx
"""

import argparse
import asyncio
import json
import os
import random

# 自动加载 .env 文件
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_SCRIPT_DIR, '..', '.env')
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _, _val = _line.partition('=')
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_HC3 = os.path.join(SCRIPT_DIR, 'training_data', 'hc3_all.jsonl')
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'output', 'ai_generated')
DEFAULT_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, 'ai_generated_v1.jsonl')

# ─── Prompt 风格模板 ───

STYLE_TEMPLATES = {
    'direct': '{question}',
    'academic': '请以学术论文的风格，严谨、专业地回答以下问题：\n\n{question}',
    'zhihu': '请以知乎高质量回答的风格回答以下问题，语言自然、有个人见解：\n\n{question}',
    'news': '请以新闻报道的风格回答以下问题，客观、简洁、信息量大：\n\n{question}',
    'creative': '请以文学创作的风格回答以下问题，语言生动、有画面感：\n\n{question}',
}

# 每种风格生成的回答数量
STYLE_GENERATION_COUNT = {
    'direct': 1,
    'academic': 1,
    'zhihu': 1,
    'news': 1,
    'creative': 1,
}


# ─── API 客户端（统一使用 openai SDK，同步调用）───

class QwenClient:
    """阿里云 DashScope Qwen API 客户端。"""

    def __init__(self, api_key, model='qwen3.6-plus'):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

    def generate(self, prompt: str) -> dict:
        """调用 Qwen API 生成文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            extra_body={'enable_thinking': False},
        )
        content = resp.choices[0].message.content or ''
        return {
            'text': content.strip(),
            'model': self.model,
            'usage': {
                'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                'total_tokens': resp.usage.total_tokens if resp.usage else 0,
            },
        }


class DeepSeekClient:
    """DeepSeek API 客户端。"""

    def __init__(self, api_key, model='deepseek-chat'):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://api.deepseek.com/v1',
        )

    def generate(self, prompt: str) -> dict:
        """调用 DeepSeek API 生成文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
        )
        content = resp.choices[0].message.content or ''
        return {
            'text': content.strip(),
            'model': self.model,
            'usage': {
                'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                'total_tokens': resp.usage.total_tokens if resp.usage else 0,
            },
        }


class OllamaClient:
    """Ollama 本地模型 API 客户端。

    Ollama 兼容 OpenAI API，直接用 OpenAI SDK 连接。
    """

    def __init__(self, host='http://localhost:11434', model='qwen2.5:7b'):
        from openai import OpenAI
        self.model = model
        self.host = host.rstrip('/')
        self.client = OpenAI(
            api_key='ollama',  # Ollama 不需要 key
            base_url=f'{self.host}/v1',
        )

    def generate(self, prompt: str) -> dict:
        """调用 Ollama 生成文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
        )
        content = resp.choices[0].message.content or ''
        return {
            'text': content.strip(),
            'model': self.model,
            'usage': {
                'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                'total_tokens': resp.usage.total_tokens if resp.usage else 0,
            },
        }

    def check_connection(self):
        """检查 Ollama 服务是否在线。"""
        try:
            import urllib.request
            req = urllib.request.Request(f'{self.host}/api/tags')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                models = [m['name'] for m in data.get('models', [])]
                return True, models
        except Exception:
            return False, []


def create_client(api_type: str, api_key: str = '', model: str = None,
                  host: str = None):
    """创建 API 客户端。"""
    if api_type == 'qwen':
        return QwenClient(api_key, model=model or 'qwen3.6-plus')
    elif api_type == 'deepseek':
        return DeepSeekClient(api_key, model=model or 'deepseek-chat')
    elif api_type == 'ollama':
        return OllamaClient(host=host or 'http://localhost:11434',
                            model=model or 'qwen2.5:7b')
    else:
        raise ValueError(f'不支持的 API 类型: {api_type}，可选: qwen, deepseek, ollama')


# ─── 数据加载 ───

def load_hc3_questions(filepath: str, min_cn_chars: int = 20):
    """从 HC3-Chinese JSONL 加载所有问题。

    Returns:
        list of dict: [{'id': int, 'question': str, 'source': str}, ...]
    """
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                q = row.get('question', '')
                if not q:
                    continue
                cn_count = sum(1 for c in q if '\u4e00' <= c <= '\u9fff')
                if cn_count < min_cn_chars:
                    continue
                questions.append({
                    'id': idx,
                    'question': q.strip(),
                    'source': row.get('source', 'unknown'),
                })
            except json.JSONDecodeError:
                continue

    print(f'[数据] 从 {filepath} 加载 {len(questions)} 条有效问题')
    return questions


def load_processed_ids(output_path: str) -> set:
    """加载已处理的 question ID 集合（用于断点续传）。
    
    同时扫描 training_data/ 下所有 jsonl 文件，统计所有已回答的 (question_id, style) 对。
    """
    processed = set()

    # 扫描 training_data/ 下所有 jsonl
    data_dir = os.path.dirname(output_path)
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith('.jsonl'):
                fpath = os.path.join(data_dir, fname)
                _load_ids_from_file(fpath, processed)

    return processed


def _load_ids_from_file(filepath: str, processed: set):
    """从单个 jsonl 文件加载已处理的 ID。"""
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                qid = row.get('question_id')
                style = row.get('style')
                if qid is not None and style is not None:
                    processed.add((qid, style))
            except (json.JSONDecodeError, KeyError):
                continue


# ─── 生成逻辑（同步 + 线程池）───

def clean_ai_response(text: str) -> str:
    """清洗 AI 模型返回的文本，去除 markdown 格式和非中文字符。

    目的：去除 AI 模型的格式特征，防止训练数据泄露标签。
    """
    import re

    # 1. 去除 markdown 标题 (# ## ### 等)
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)

    # 2. 去除 markdown 粗体/斜体 (**bold**, *italic*, ___bold___)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)

    # 3. 去除 markdown 列表标记 (- * + 1. 2. 等)
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+[.)]\s+', '', text, flags=re.MULTILINE)

    # 4. 去除 markdown 代码块 (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 5. 去除行内代码 (`code`)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # 6. 去除 markdown 链接 [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 7. 去除 markdown 表格 (|...|)
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(r'^\s*\|', l)]
    text = '\n'.join(lines)

    # 8. 去除水平分割线 (--- *** ___)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 9. 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 10. 去除 emoji 和特殊符号
    import unicodedata
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # 去除 emoji 范围
        if (0x1F600 <= cp <= 0x1F64F or   # emoticons
            0x1F300 <= cp <= 0x1F5FF or   # misc symbols
            0x1F680 <= cp <= 0x1F6FF or   # transport
            0x1F1E0 <= cp <= 0x1F1FF or   # flags
            0x2600 <= cp <= 0x26FF or     # misc
            0x2700 <= cp <= 0x27BF or     # dingbats
            0xFE00 <= cp <= 0xFE0F or     # variation selectors
            0x1F900 <= cp <= 0x1F9FF or   # supplemental
            0x1FA00 <= cp <= 0x1FA6F or   # chess
            0x1FA70 <= cp <= 0x1FAFF or   # symbols extended
            0x200D == cp or               # zero-width joiner
            0x20E3 == cp or               # combining enclosing keycap
            0xE0020 <= cp <= 0xE007F):    # tags
            continue
        # 去除其他非中文、非标点、非基本符号的字符
        if unicodedata.category(ch).startswith('So') and ch not in '。，！？、；：""''（）—…·':
            continue
        cleaned.append(ch)
    text = ''.join(cleaned)

    # 11. 去除多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def generate_one(
    client,
    question_id: int,
    question: str,
    style: str,
    max_retries: int = 3,
):
    """生成一条 AI 回答（带重试）。"""
    template = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES['direct'])
    prompt = template.format(question=question)
    t0 = time.time()

    for attempt in range(max_retries):
        try:
            result = client.generate(prompt)
            elapsed = time.time() - t0
            cleaned_text = clean_ai_response(result['text'])
            return {
                'question_id': question_id,
                'question': question,
                'ai_answer': cleaned_text,
                'model': result['model'],
                'style': style,
                'generated_at': datetime.now().isoformat(),
                'usage': result.get('usage', {}),
                '_elapsed': round(elapsed, 1),
            }
        except Exception as e:
            err_msg = f'{type(e).__name__}: {e}' if str(e) else f'{type(e).__name__}'
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.random()
                print(f'  [重试] question_id={question_id} style={style} '
                      f'attempt={attempt+1} error={err_msg}，等待 {wait:.1f}s')
                time.sleep(wait)
            else:
                print(f'  [失败] question_id={question_id} style={style} '
                      f'error={err_msg}，已重试 {max_retries} 次')
                return None


def generate_all(
    client,
    questions: list,
    output_path: str,
    concurrency: int = 5,
    resume: bool = True,
    resume_from: str = '',
    quick: bool = False,
):
    """批量生成所有 AI 回答（使用线程池并发）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 加载已处理 ID（自动扫描 training_data/ 下所有 jsonl）
    processed = set()
    if resume:
        processed = load_processed_ids(output_path)
    print(f'[续传] 已处理 {len(processed)} 条记录')

    # 构建任务列表
    tasks = []
    for q in questions:
        if quick and q['id'] >= 10:
            break
        for style, count in STYLE_GENERATION_COUNT.items():
            for _ in range(count):
                key = (q['id'], style)
                if key in processed:
                    continue
                tasks.append((q['id'], q['question'], style))

    if not tasks:
        print('[完成] 所有任务已处理，无需新生成')
        return

    print(f'[任务] 待生成 {len(tasks)} 条 AI 回答')
    print(f'[配置] 并发={concurrency}, 模型={client.model}')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_generated = 0
    total_failed = 0
    start_time = time.time()
    last_batch_time = start_time
    last_batch_count = 0
    recent_elapses = []  # 最近 N 条的请求耗时，用于计算并发效率

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(generate_one, client, qid, q, style): (qid, q, style)
            for qid, q, style in tasks
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                total_generated += 1
                # 记录耗时（去掉内部字段）
                req_elapsed = result.pop('_elapsed', None)
                if req_elapsed is not None:
                    recent_elapses.append(req_elapsed)
                    if len(recent_elapses) > 50:
                        recent_elapses.pop(0)
            else:
                total_failed += 1

            # 进度报告（每 5 条）
            if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                now = time.time()
                batch_count = total_generated - last_batch_count
                batch_elapsed = now - last_batch_time
                batch_rate = batch_count / batch_elapsed if batch_elapsed > 0 else 0
                # 计算平均请求耗时和有效并发
                avg_req_time = sum(recent_elapses) / len(recent_elapses) if recent_elapses else 0
                effective_concurrency = batch_rate * avg_req_time if avg_req_time > 0 else 0
                print(f'  [进度] {i+1}/{len(tasks)} | '
                      f'成功={total_generated} 失败={total_failed} | '
                      f'速率={batch_rate:.1f}条/s | '
                      f'有效并发≈{effective_concurrency:.1f} | '
                      f'平均耗时={avg_req_time:.1f}s/条 | '
                      f'本批耗时={batch_elapsed:.0f}s')
                last_batch_time = now
                last_batch_count = total_generated

    elapsed = time.time() - start_time
    print(f'\n[完成] 生成 {total_generated} 条，失败 {total_failed} 条')
    print(f'[耗时] {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'[输出] {output_path}')


# ─── CLI ───

def parse_args():
    parser = argparse.ArgumentParser(
        description='AI 文本语料生成脚本 — 用 Qwen/DeepSeek/Ollama API 生成多种风格 AI 回答',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--api', type=str, default='ollama',
                        choices=['qwen', 'deepseek', 'ollama'],
                        help='API 类型 (默认: ollama)')
    parser.add_argument('--api-key', type=str, default='',
                        help='API Key（qwen/deepseek 需要，ollama 不需要）')
    parser.add_argument('--host', type=str, default='http://192.168.0.188:11434',
                        help='Ollama 服务地址 (默认: http://192.168.0.188:11434)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型名称 (默认: qwen3.6-plus / deepseek-chat / qwen2.5:7b)')
    parser.add_argument('--hc3', type=str, default=DEFAULT_HC3,
                        help=f'HC3 数据路径 (默认: {DEFAULT_HC3})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'输出文件路径 (默认: {DEFAULT_OUTPUT})')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='并发请求数 (默认: 3，Ollama 本地建议 1-3)')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用断点续传，重新生成所有数据')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式，只生成 10 条问题')
    return parser.parse_args()


def main():
    args = parse_args()

    # API Key: Ollama 不需要，云端 API 需要
    api_key = args.api_key
    if args.api != 'ollama' and not api_key:
        # 尝试多个环境变量名
        env_vars = {
            'qwen': ['DASHSCOPE_API_KEY', 'QWEN_API_KEY'],
            'deepseek': ['DEEPSEEK_API_KEY'],
        }
        for ev in env_vars.get(args.api, []):
            api_key = os.environ.get(ev, '')
            if api_key:
                break
    if args.api != 'ollama' and not api_key:
        print(f'错误: {args.api} 需要 API Key。请通过 --api-key 参数或环境变量设置。')
        sys.exit(1)

    # 默认模型名
    default_models = {
        'qwen': 'qwen3.6-plus',
        'deepseek': 'deepseek-chat',
        'ollama': 'qwen2.5:7b',
    }
    model_display = args.model or default_models[args.api]

    print(f'{"="*60}')
    print('AI 文本语料生成脚本')
    # 如果用户没有指定输出文件，自动加时间戳
    if args.output == DEFAULT_OUTPUT:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(DEFAULT_OUTPUT_DIR,
                                    f'ai_generated_{args.api}_{model_display}_{ts}.jsonl')

    print(f'{"="*60}')
    print(f'  API:    {args.api}')
    print(f'  模型:   {model_display}')
    if args.api == 'ollama':
        print(f'  地址:   {args.host}')
    print(f'  HC3:    {args.hc3}')
    print(f'  输出:   {args.output}')
    print(f'  并发:   {args.concurrency}')
    print(f'  快速:   {args.quick}')
    print(f'  续传:   {not args.no_resume}')
    print(f'{"="*60}\n')

    # Ollama: 先检查连接
    if args.api == 'ollama':
        client = create_client(args.api, '', args.model, args.host)
        print('[检查] 正在连接 Ollama 服务...')
        ok, models = client.check_connection()
        if ok:
            print(f'[✅] Ollama 在线，可用模型: {models}')
            if model_display not in models:
                print(f'[⚠️]  警告: 模型 "{model_display}" 不在列表中，将尝试使用')
        else:
            print(f'[❌] 无法连接到 Ollama 服务 ({args.host})，请确认服务已启动')
            sys.exit(1)
        print()
    else:
        client = create_client(args.api, api_key, args.model)

    # 加载问题
    questions = load_hc3_questions(args.hc3)
    if not questions:
        print('错误: 未找到有效问题，请检查 HC3 数据路径。')
        sys.exit(1)

    # 开始生成
    generate_all(
        client=client,
        questions=questions,
        output_path=args.output,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        quick=args.quick,
    )


if __name__ == '__main__':
    main()

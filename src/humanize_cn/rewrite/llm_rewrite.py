"""LLM API 改写模块 — 支持多 API 后端（DeepSeek / OpenAI / 兼容接口）

通过调用外部 LLM API 对高风险段落进行自然语言改写，
从根本上解决静态规则过拟合问题。

支持的 API 后端：
- DeepSeek (https://api.deepseek.com)
- OpenAI (https://api.openai.com)
- 任何兼容 OpenAI Chat Completions API 的服务

配置方式：
1. 环境变量：DEEPSEEK_API_KEY / OPENAI_API_KEY
2. 配置文件：config.json 的 [llm] section
3. 运行时参数：humanize_academic(text, llm_provider='deepseek')
"""

import os
import json
import time
import logging

logger = logging.getLogger(__name__)

# 加载 .env 文件
def _load_dotenv():
    """加载项目根目录的 .env 文件。"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value and key not in os.environ:
                        os.environ[key] = value

_load_dotenv()

# Lazy import httpx
_httpx = None


def _get_httpx():
    global _httpx
    if _httpx is None:
        try:
            import httpx
            _httpx = httpx
        except ImportError:
            raise ImportError("需要安装 httpx: pip install httpx")
    return _httpx


# ═══════════════════════════════════════════════════════════════════
#  API 配置
# ═══════════════════════════════════════════════════════════════════

# API 端点配置
_API_ENDPOINTS = {
    'deepseek': {
        'base_url': 'https://api.deepseek.com',
        'chat_path': '/v1/chat/completions',
        'env_key': 'DEEPSEEK_API_KEY',
        'default_model': 'deepseek-chat',
    },
    'openai': {
        'base_url': 'https://api.openai.com',
        'chat_path': '/v1/chat/completions',
        'env_key': 'OPENAI_API_KEY',
        'default_model': 'gpt-4o-mini',
    },
    'dashscope': {
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'chat_path': '/chat/completions',
        'env_key': 'DASHSCOPE_API_KEY',
        'default_model': 'qwen3.5-397b-a17b',
    },
}

# 学术改写 Prompt — 基于 SKILL.md v3.0 改写策略
_ACADEMIC_REWRITE_PROMPT = """你是一位经验丰富的中文学术编辑，擅长降低论文的 AIGC 检测分数。请对以下论文段落进行改写。

【改写目标】让这段文字通过知网/维普/万方的 AIGC 检测，同时保持学术准确性和原意。

【第一步：砍掉 AI 套路】
- 去掉"首先…其次…最后"三段式，用自然过渡代替
- 删掉机械连接词：值得注意的是、综上所述、不难发现、总而言之、与此同时、由此可见、不仅如此、换句话说、更重要的是、不可否认、显而易见、不言而喻、归根结底
- 删掉空洞宏大词：赋能、闭环、数字化转型、协同增效、降本增效、深度融合、全方位、多维度、系统性、高质量发展、新质生产力
- 删掉 AI 高频词：助力、彰显、凸显、底层逻辑、抓手、触达、沉淀、复盘、迭代、破圈、颠覆
- 删掉模板句式：随着…的不断发展、在当今…时代、在…的背景下、作为…的重要组成部分、这不仅…更是…
- 删掉平衡论述套话：虽然…但是…同时、既有…也有…更有

【第二步：替换学术套话（保持学术性）】
- 本文旨在 → 本研究聚焦于 / 本文尝试 / 本研究关注
- 具有重要意义 → 值得关注 / 有一定参考价值
- 研究表明 → 前人研究发现 / 已有文献显示 / 笔者观察到
- 进行了深入分析 → 做了初步探讨 / 展开了讨论
- 取得了显著成效 → 产生了一定效果 / 初见成效
- 被广泛应用 → 得到较多运用 / 在多个领域有所应用
- 被认为是 → 通常被看作 / 一般认为
- 近年来 → 过去数年间 / 此前数年
- 首先/其次/最后 → 其一/其二/末了（如果必须列举）
- 因此 → 故而 / 由此 / 据此

【第三步：句式重组】
- 打破均匀的句式节奏：长短句交替，偶尔用短句（5-10字）做强调
- 过长的句子在"但是""不过""同时"等转折处断开
- 不要让每句话长度差不多
- 不要使用排比句式

【第四步：减少重复用词】
同一个词出现 3 次以上就换同义词。比如"进行"换成"做""开展""着手"。

【第五步：注入人味（学术场景适度）】
- 增强作者主体性：笔者认为、本研究发现、笔者倾向于认为
- 注入适度学术犹豫语：可能、在一定程度上、就目前而言、初步来看
- 偶尔用具体的表述代替抽象概括

【第六步：保持不变】
- 专业术语一个字不改（STM32、ESP8266、Django、ORM、WiFi、UART、ADC、I2C、SPI、DMA、TCP/IP、HTTP、JSON、MQTT 等所有技术名词）
- 数据、公式、引用标记 [1][2][3] 不变
- 不要添加新观点或信息
- 不要删除重要内容

【绝对不要】
- 不要输出任何解释，只输出改写后的段落
- 不要使用"值得注意的是""需要指出的是""综上所述"等 AI 高频短语
- 不要让每句话长度差不多
- 不要使用排比句式

原文：
{text}

改写后："""

# 句子级精准改写 Prompt（用于 perplexity 定向改写）
_SENTENCE_REWRITE_PROMPT = """请对以下句子进行改写，使其读起来更自然，像人类写的。保持原意不变，保持专业术语不变。

原文：{sentence}

改写后："""


# ═══════════════════════════════════════════════════════════════════
#  API 调用
# ═══════════════════════════════════════════════════════════════════

def _call_chat_api(provider, api_key, model, messages, temperature=0.7,
                   max_tokens=2048, timeout=60, base_url=None, extra_body=None):
    """调用 Chat Completions API。

    Args:
        provider: API 提供商 ('deepseek' / 'openai' / 'dashscope')
        api_key: API Key
        model: 模型名称
        messages: 消息列表 [{"role": "user", "content": "..."}]
        temperature: 采样温度
        max_tokens: 最大输出 token 数
        timeout: 超时秒数
        base_url: 自定义 API 端点（覆盖默认）
        extra_body: 额外的请求体参数（如 {"enable_thinking": False}）
    Returns:
        str: 模型回复内容
    Raises:
        Exception: API 调用失败
    """
    httpx = _get_httpx()

    if base_url is None:
        config = _API_ENDPOINTS.get(provider, _API_ENDPOINTS['dashscope'])
        base_url = config['base_url']
        chat_path = config['chat_path']
    else:
        chat_path = '/v1/chat/completions'

    url = base_url.rstrip('/') + chat_path

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

    # 合并 extra_body（如关闭深度思考）
    if extra_body:
        payload.update(extra_body)

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content'].strip()
    except httpx.TimeoutException:
        raise Exception(f'API 超时 ({timeout}s)')
    except httpx.HTTPStatusError as e:
        raise Exception(f'API 错误: {e.response.status_code} {e.response.text[:200]}')
    except (KeyError, IndexError):
        raise Exception(f'API 返回格式异常: {resp.text[:200]}')


def _get_api_config(provider=None):
    """获取 API 配置（Key、Model、Base URL）。

    优先级：参数 > 环境变量 > 配置文件 > 默认值

    Returns:
        (provider, api_key, model, base_url) 或 None（如果未配置）
    """
    if provider is None:
        # 按优先级尝试各 API
        for p in ['dashscope', 'deepseek', 'openai']:
            config = _get_api_config(p)
            if config is not None:
                return config
        return None

    config = _API_ENDPOINTS.get(provider)
    if config is None:
        logger.warning(f'未知的 API 提供商: {provider}')
        return None

    # 获取 API Key
    api_key = os.environ.get(config['env_key'], '')
    if not api_key:
        # 尝试通用环境变量
        api_key = os.environ.get('LLM_API_KEY', '')
    if not api_key:
        return None

    # 获取模型
    model = os.environ.get(f'{provider.upper()}_MODEL', config['default_model'])

    # 获取自定义 base_url
    base_url = os.environ.get(f'{provider.upper()}_BASE_URL', None)

    return (provider, api_key, model, base_url)


# ═══════════════════════════════════════════════════════════════════
#  改写函数
# ═══════════════════════════════════════════════════════════════════

def llm_rewrite_paragraph(text, provider=None, temperature=0.7, max_tokens=2048,
                         detect_fn=None, score_fn=None, max_retries=2):
    """使用 LLM API 改写段落，带反馈闭环。

    如果提供了 detect_fn 和 score_fn，会在改写后检测分数：
    - 分数下降 → 接受改写
    - 分数不变或上升 → 提高温度重试，最多 max_retries 次
    - 所有尝试都失败 → 回退到本地规则改写后的版本（即输入的 text）

    Args:
        text: 待改写的段落文本
        provider: API 提供商 ('dashscope' / 'deepseek' / 'openai' / None=自动选择)
        temperature: 采样温度（0.0-1.0，越高越随机）
        max_tokens: 最大输出 token 数
        detect_fn: 检测函数 (text) -> (issues, metrics)，可选
        score_fn: 评分函数 (issues, metrics) -> int，可选
        max_retries: 最大重试次数（每次提高温度）
    Returns:
        str: 改写后的文本，如果 API 不可用则返回原文
    """
    config = _get_api_config(provider)
    if config is None:
        logger.warning('LLM API 未配置，跳过 LLM 改写')
        return text

    provider, api_key, model, base_url = config

    # 如果有检测函数，先计算原始分数
    original_score = None
    if detect_fn and score_fn:
        try:
            orig_issues, orig_metrics = detect_fn(text)
            original_score = score_fn(orig_issues, orig_metrics)
        except Exception:
            pass

    messages = [
        {'role': 'user', 'content': _ACADEMIC_REWRITE_PROMPT.format(text=text)}
    ]

    # DashScope Qwen 关闭深度思考
    extra_body = None
    if provider == 'dashscope':
        extra_body = {'enable_thinking': False}

    best_text = text
    best_score = original_score

    for attempt in range(max_retries + 1):
        current_temp = min(temperature + attempt * 0.15, 1.2)

        try:
            result = _call_chat_api(
                provider, api_key, model, messages,
                temperature=current_temp, max_tokens=max_tokens,
                base_url=base_url, extra_body=extra_body,
            )
            # 清理输出
            result = result.strip()
            for prefix in ['改写后：', '改写后:', '改写：', '改写:']:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()

            # 如果没有检测函数，直接返回第一次改写结果
            if not detect_fn or not score_fn:
                return result

            # 检测改写后分数
            try:
                new_issues, new_metrics = detect_fn(result)
                new_score = score_fn(new_issues, new_metrics)
            except Exception:
                new_score = None

            if new_score is not None and (best_score is None or new_score < best_score):
                best_score = new_score
                best_text = result
                if original_score is not None and new_score < original_score:
                    logger.info(f'LLM 改写成功: {original_score} → {new_score} (降幅 {original_score - new_score})')
                    break  # 分数下降，接受
                else:
                    logger.debug(f'第{attempt + 1}次: 分数 {original_score} → {new_score}，继续尝试')
            else:
                # 检测失败，保留当前最佳
                if best_text == text:
                    best_text = result  # 至少用第一次改写结果

        except Exception as e:
            logger.warning(f'LLM 改写第{attempt + 1}次失败: {e}')
            continue

    return best_text


def llm_rewrite_sentences(text, provider=None, temperature=0.7, max_sentences=5):
    """使用 LLM API 对低 perplexity 句子进行精准改写。

    1. 逐句计算 perplexity
    2. 选出 perplexity 最低的 N 个句子
    3. 对每个句子调用 LLM 改写
    4. 用 BERT scorer 验证改写质量

    Args:
        text: 待改写的段落文本
        provider: API 提供商
        temperature: 采样温度
        max_sentences: 最多改写几个句子
    Returns:
        str: 改写后的文本
    """
    config = _get_api_config(provider)
    if config is None:
        return text

    provider, api_key, model, base_url = config

    # 延迟导入避免循环依赖
    try:
        from .targeted import _split_sentences, _get_sentence_perplexity
    except ImportError:
        from humanize_cn.rewrite.targeted import _split_sentences, _get_sentence_perplexity

    sentences = _split_sentences(text)
    if len(sentences) < 2:
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
        if rewritten >= max_sentences:
            break
        if ppl <= 0:
            continue

        # 跳过太短的句子
        cn_count = sum(1 for c in sent if '\u4e00' <= c <= '\u9fff')
        if cn_count < 15:
            continue

        # 跳过包含引用标记的句子（保持引用完整）
        if '[0]' in sent or '参考文献' in sent:
            continue

        messages = [
            {'role': 'user', 'content': _SENTENCE_REWRITE_PROMPT.format(sentence=sent)}
        ]

        extra_body = {'enable_thinking': False} if provider == 'dashscope' else None

        try:
            new_sent = _call_chat_api(
                provider, api_key, model, messages,
                temperature=temperature, max_tokens=512,
                base_url=base_url, extra_body=extra_body,
            )
            new_sent = new_sent.strip()
            for prefix in ['改写后：', '改写后:', '改写：', '改写:']:
                if new_sent.startswith(prefix):
                    new_sent = new_sent[len(prefix):].strip()

            # 验证：perplexity 确实提升
            new_ppl = _get_sentence_perplexity(new_sent)
            if new_ppl > ppl * 1.05:  # 至少提升 5%
                sentences[idx] = new_sent
                rewritten += 1
                logger.info(f'句子改写成功: ppl {ppl:.1f} → {new_ppl:.1f}')
            else:
                logger.debug(f'句子改写未通过验证: ppl {ppl:.1f} → {new_ppl:.1f}')

            # 避免请求过快
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f'句子改写失败: {e}')
            continue

    return ''.join(sentences)


def llm_feedback_rewrite(text, detect_fn, score_fn, provider=None,
                         max_rounds=2, target_drop=10):
    """LLM 反馈闭环改写：改写→检测→调整循环。

    Args:
        text: 输入文本
        detect_fn: 检测函数 (text) -> (issues, metrics)
        score_fn: 评分函数 (issues, metrics) -> int
        provider: API 提供商
        max_rounds: 最大迭代轮数
        target_drop: 目标降幅
    Returns:
        (best_text, original_score, best_score, drop)
    """
    original_issues, original_metrics = detect_fn(text)
    original_score = score_fn(original_issues, original_metrics)

    best_text = text
    best_score = original_score

    for round_num in range(max_rounds):
        if best_score <= 0:
            break
        if original_score - best_score >= target_drop:
            break

        # 调用 LLM 改写
        candidate = llm_rewrite_paragraph(best_text, provider=provider,
                                          temperature=0.7 + round_num * 0.1)

        if candidate == best_text:
            break

        # 检测改写后分数
        new_issues, new_metrics = detect_fn(candidate)
        new_score = score_fn(new_issues, new_metrics)

        if new_score < best_score:
            best_text = candidate
            best_score = new_score
            logger.info(f'第{round_num + 1}轮: 分数 {best_score + (original_score - best_score)} → {best_score}')
        else:
            logger.debug(f'第{round_num + 1}轮: 分数未下降 ({best_score} → {new_score})，回退')

    drop = original_score - best_score
    return best_text, original_score, best_score, drop


def is_llm_available(provider=None):
    """检查 LLM API 是否可用。"""
    config = _get_api_config(provider)
    return config is not None

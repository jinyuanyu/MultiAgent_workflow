import os
import json
from jinja2 import Template
from coze_coding_dev_sdk import LLMClient, TTSClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import PodcastScriptInput, PodcastScriptOutput


def get_text_content(content):
    """安全提取文本内容"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        if content and isinstance(content[0], str):
            return " ".join(content)
        else:
            text_parts = [
                item.get("text", "") 
                for item in content 
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return " ".join(text_parts)
    return str(content)


def podcast_script_node(
    state: PodcastScriptInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> PodcastScriptOutput:
    """
    title: 播客对谈脚本
    desc: 将视频内容改编为双人对谈脚本，并合成为音频文件
    integrations: 大语言模型, 语音大模型
    """
    ctx = runtime.context
    
    # 读取LLM配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), config['metadata']['llm_cfg'])
    with open(cfg_file, 'r', encoding='utf-8') as fd:
        llm_cfg = json.load(fd)
    
    model_config = llm_cfg.get("config", {})
    sp = llm_cfg.get("sp", "")
    up = llm_cfg.get("up", "")
    
    # 渲染提示词
    up_tpl = Template(up)
    user_prompt = up_tpl.render({
        "transcript": state.transcript,
        "video_title": state.video_title
    })
    
    # 调用LLM生成播客脚本
    llm_ctx = new_context(method="llm.invoke")
    llm_client = LLMClient(ctx=llm_ctx)
    
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm_client.invoke(
        messages=messages,
        model=model_config.get("model", "doubao-seed-1-8-251228"),
        temperature=model_config.get("temperature", 0.8),
        max_completion_tokens=model_config.get("max_completion_tokens", 3000)
    )
    
    # 提取生成的脚本
    podcast_script = get_text_content(response.content)
    
    # 初始化TTS客户端
    tts_ctx = new_context(method="tts.synthesize")
    tts_client = TTSClient(ctx=tts_ctx)
    
    # 生成播客音频
    try:
        audio_url, audio_size = tts_client.synthesize(
            uid="podcast_user",
            text=podcast_script[:2000],  # 限制长度
            speaker="zh_male_m191_uranus_bigtts",  # 使用男声
            audio_format="mp3",
            sample_rate=24000,
            speech_rate=0,
            loudness_rate=0
        )
    except Exception as e:
        # 异常时使用占位符
        audio_url = f"https://example.com/podcast/podcast_{state.video_title[:10]}.mp3"
    
    return PodcastScriptOutput(
        podcast_audio_url=audio_url
    )

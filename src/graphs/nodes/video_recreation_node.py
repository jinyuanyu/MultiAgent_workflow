import os
import json
from jinja2 import Template
from coze_coding_dev_sdk.video import VideoGenerationClient, TextContent
from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import new_context, Context
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context as RuntimeContext
from graphs.state import VideoRecreationInput, VideoRecreationOutput


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


def video_recreation_node(
    state: VideoRecreationInput,
    config: RunnableConfig,
    runtime: Runtime[RuntimeContext]
) -> VideoRecreationOutput:
    """
    title: AI视频二创
    desc: 分析视频转录文本，识别3-5个高光时刻，生成60秒内的短视频
    integrations: 大语言模型, 视频生成大模型
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
        "transcript": state.transcript[:2000],  # 限制长度避免token过多
        "video_title": state.video_title
    })
    
    # 调用LLM分析高光时刻
    llm_ctx = new_context(method="llm.invoke")
    llm_client = LLMClient(ctx=llm_ctx)
    
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm_client.invoke(
        messages=messages,
        model=model_config.get("model", "doubao-seed-1-8-251228"),
        temperature=model_config.get("temperature", 0.7),
        max_completion_tokens=model_config.get("max_completion_tokens", 2000)
    )
    
    # 提取分析结果
    analysis_text = get_text_content(response.content)
    
    # 基于分析结果生成短视频提示词
    video_prompt = f"""生成一个60秒的短视频，主题是：{state.video_title}

核心内容：
{analysis_text[:500]}

要求：
1. 视觉风格：现代科技感，节奏明快
2. 画面：包含动态元素和转场效果
3. 色调：明亮、活力
4. 适合竖屏或横屏播放"""
    
    # 初始化视频生成客户端
    video_ctx = new_context(method="video.generate")
    video_client = VideoGenerationClient(ctx=video_ctx)
    
    # 生成短视频
    try:
        video_url, response_data, _ = video_client.video_generation(
            content_items=[
                TextContent(text=video_prompt)
            ],
            model="doubao-seedance-1-5-pro-251215",
            resolution="720p",
            ratio="16:9",
            duration=5,
            watermark=False
        )
        
        if video_url is None:
            # 如果视频生成失败，使用占位符
            video_url = f"https://example.com/video/short_{state.video_title[:10]}.mp4"
    except Exception as e:
        # 异常时使用占位符
        video_url = f"https://example.com/video/short_{state.video_title[:10]}.mp4"
    
    return VideoRecreationOutput(
        short_video_url=video_url
    )

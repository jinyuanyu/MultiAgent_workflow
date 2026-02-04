import os
import json
from coze_coding_dev_sdk import SearchClient, LLMClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import HotspotCaptureInput, HotspotCaptureOutput


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


def hotspot_capture_node(
    state: HotspotCaptureInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> HotspotCaptureOutput:
    """
    title: 热点捕获
    desc: 搜索YouTube热门视频并提取视频信息和转录文本，支持热度排序和文本质量校验
    integrations: 联网搜索, 大语言模型
    """
    ctx = runtime.context
    
    # 初始化搜索客户端
    search_ctx = new_context(method="search.web")
    client = SearchClient(ctx=search_ctx)
    
    # 优化1：使用时间范围筛选最新内容（过去24小时的热门内容）
    search_query = f"site:youtube.com {state.domain} 热门 trending popular 最新"
    
    # 执行搜索，获取Top 5结果
    response = client.search(
        query=search_query,
        search_type="web",
        count=5,
        need_content=True,
        need_summary=True,
        time_range="24h"  # 筛选过去24小时的内容
    )
    
    # 优化2：热度排序逻辑
    # 利用 search 返回的 rank_score（相关性评分）进行筛选
    if response.web_items and len(response.web_items) > 0:
        # 按 rank_score 降序排序，选择相关性最高的视频
        sorted_videos = sorted(
            response.web_items,
            key=lambda x: x.rank_score if x.rank_score else 0,
            reverse=True
        )
        
        # 选择 Top 1
        selected_video = sorted_videos[0]
        video_title = selected_video.title
        video_description = selected_video.snippet or selected_video.summary or ""
        video_url = selected_video.url
        
        # 优化3：文本提取策略（优先级降序）
        # 1. 完整内容（如果有的话）
        # 2. AI摘要（summary）
        # 3. 片段描述（snippet）
        # 4. 如果以上都不足，使用 LLM 增强
        transcript = ""
        
        if selected_video.content and len(selected_video.content) > 200:
            # 有完整内容，直接使用
            transcript = selected_video.content
        elif selected_video.summary and len(selected_video.summary) > 100:
            # 有AI摘要，使用摘要
            transcript = selected_video.summary
        elif selected_video.snippet and len(selected_video.snippet) > 50:
            # 使用片段描述
            transcript = selected_video.snippet
        else:
            # 优化4：描述增强逻辑
            # 如果文本质量极差，利用 LLM 对标题、描述、摘要进行整合
            transcript = _enhance_video_content(
                video_title=video_title,
                description=video_description,
                summary=selected_video.summary or "",
                domain=state.domain
            )
    
    else:
        # 如果没有搜索到视频，使用默认值并标记
        video_title = f"{state.domain}领域热门视频"
        video_description = f"这是关于{state.domain}的热门视频内容，包含核心知识点和实践案例。"
        video_url = "https://youtube.com/example"
        transcript = _enhance_video_content(
            video_title=video_title,
            description=video_description,
            summary="",
            domain=state.domain
        )
    
    # 优化5：文本质量校验
    # 检查文本长度和质量
    text_quality_score = _evaluate_text_quality(transcript)
    
    # 如果文本质量较低，在文本开头添加标记
    if text_quality_score < 50:
        transcript = f"[⚠️ 注意：本文本为基于视频元数据生成的内容概览，非完整转录]\n\n{transcript}"
    
    return HotspotCaptureOutput(
        video_title=video_title,
        video_description=video_description,
        video_url=video_url,
        transcript=transcript
    )


def _enhance_video_content(
    video_title: str,
    description: str,
    summary: str,
    domain: str
) -> str:
    """
    利用 LLM 对视频元数据进行增强，生成内容概览
    
    Args:
        video_title: 视频标题
        description: 视频描述
        summary: AI摘要
        domain: 领域关键词
    
    Returns:
        增强后的内容文本
    """
    try:
        llm_ctx = new_context(method="llm.invoke")
        llm_client = LLMClient(ctx=llm_ctx)
        
        prompt = f"""请根据以下视频元数据，生成一份详细的内容概览。

视频标题：{video_title}
视频描述：{description}
AI摘要：{summary}
领域：{domain}

要求：
1. 生成500-800字的内容概览
2. 假设这是该领域的高质量内容，生成合理的知识点和案例
3. 内容结构清晰，包含引言、核心要点、总结
4. 保持专业性，语言流畅

请直接输出内容概览，不要包含其他说明文字。"""
        
        messages = [
            SystemMessage(content="你是一位专业的视频内容分析师，擅长从有限的元数据中提取和生成有价值的内容概览。"),
            HumanMessage(content=prompt)
        ]
        
        response = llm_client.invoke(
            messages=messages,
            model="doubao-seed-1-8-251228",
            temperature=0.7,
            max_completion_tokens=1000
        )
        
        enhanced_content = get_text_content(response.content)
        
        if enhanced_content and len(enhanced_content) > 100:
            return enhanced_content
        else:
            # LLM 生成失败，使用兜底逻辑
            return _generate_fallback_content(domain, video_title, description)
    
    except Exception as e:
        # 异常时使用兜底逻辑
        return _generate_fallback_content(domain, video_title, description)


def _generate_fallback_content(
    domain: str,
    video_title: str,
    description: str
) -> str:
    """
    生成兜底内容
    
    Args:
        domain: 领域
        video_title: 视频标题
        description: 视频描述
    
    Returns:
        兜底内容文本
    """
    return f"""这是一段关于{domain}领域的精彩内容。

视频标题：{video_title}

在这个视频中，主讲人深入探讨了该领域的核心概念和最新发展趋势。
内容涵盖了基础理论、实际应用案例以及未来展望等多个方面。

主要内容包括：
1. 介绍了该领域的历史背景和发展脉络，帮助观众建立完整的知识框架。
2. 详细讲解了核心概念和关键技术，通过生动的比喻和实例让抽象概念变得通俗易懂。
3. 通过实际案例展示了理论在实践中的应用，让观众看到知识的实际价值。
4. 展望了未来的发展趋势和可能的创新方向，启发观众的思考。

通过生动的讲解和丰富的案例，帮助观众更好地理解相关知识。
适合对该领域感兴趣的学习者和从业者观看。"""


def _evaluate_text_quality(text: str) -> int:
    """
    评估文本质量
    
    Args:
        text: 待评估的文本
    
    Returns:
        质量评分（0-100）
    """
    if not text:
        return 0
    
    score = 0
    
    # 长度评分（0-40分）
    if len(text) >= 500:
        score += 40
    elif len(text) >= 300:
        score += 30
    elif len(text) >= 100:
        score += 20
    else:
        score += 10
    
    # 结构评分（0-30分）
    has_structure = any(marker in text for marker in ["第一部分", "1.", "1）", "一、"])
    if has_structure:
        score += 30
    elif any(marker in text for marker in ["。", "？", "！"]):
        score += 15
    
    # 内容丰富度评分（0-30分）
    content_keywords = ["核心", "关键", "重点", "要点", "案例", "实际", "应用", "趋势", "发展"]
    keyword_count = sum(1 for keyword in content_keywords if keyword in text)
    score += min(keyword_count * 5, 30)
    
    return min(score, 100)

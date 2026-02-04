import os
import json
from jinja2 import Template
from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import LearningGuideInput, LearningGuideOutput


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


def learning_guide_node(
    state: LearningGuideInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> LearningGuideOutput:
    """
    title: 深度学习指南
    desc: 基于视频内容生成结构化的学习指南Markdown文档
    integrations: 大语言模型
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
    
    # 调用LLM生成学习指南
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
        max_completion_tokens=model_config.get("max_completion_tokens", 4096)
    )
    
    # 提取生成的Markdown内容
    learning_guide = get_text_content(response.content)
    
    return LearningGuideOutput(
        learning_guide=learning_guide
    )

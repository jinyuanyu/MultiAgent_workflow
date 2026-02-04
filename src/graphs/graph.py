from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput
)
from graphs.nodes.hotspot_capture_node import hotspot_capture_node
from graphs.nodes.video_recreation_node import video_recreation_node
from graphs.nodes.learning_guide_node import learning_guide_node
from graphs.nodes.podcast_script_node import podcast_script_node
from graphs.nodes.result_summary_node import result_summary_node


# 创建状态图，指定工作流的入参和出参
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

# 添加节点
# 热点捕获节点
builder.add_node("hotspot_capture", hotspot_capture_node)

# AI视频二创节点（Agent节点，使用LLM）
builder.add_node(
    "video_recreation",
    video_recreation_node,
    metadata={"type": "agent", "llm_cfg": "config/video_recreation_cfg.json"}
)

# 深度学习指南节点（Agent节点，使用LLM）
builder.add_node(
    "learning_guide",
    learning_guide_node,
    metadata={"type": "agent", "llm_cfg": "config/learning_guide_cfg.json"}
)

# 播客对谈脚本节点（Agent节点，使用LLM）
builder.add_node(
    "podcast_script",
    podcast_script_node,
    metadata={"type": "agent", "llm_cfg": "config/podcast_script_cfg.json"}
)

# 结果汇总节点
builder.add_node("result_summary", result_summary_node)

# 设置入口点
builder.set_entry_point("hotspot_capture")

# 添加边
# 热点捕获后，并行执行三个任务
builder.add_edge("hotspot_capture", "video_recreation")
builder.add_edge("hotspot_capture", "learning_guide")
builder.add_edge("hotspot_capture", "podcast_script")

# 三个并行任务都完成后，汇聚到结果汇总节点
builder.add_edge(
    ["video_recreation", "learning_guide", "podcast_script"],
    "result_summary"
)

# 结果汇总后结束
builder.add_edge("result_summary", END)

# 编译图
main_graph = builder.compile()

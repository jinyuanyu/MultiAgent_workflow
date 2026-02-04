from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ResultSummaryInput, ResultSummaryOutput


def result_summary_node(
    state: ResultSummaryInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ResultSummaryOutput:
    """
    title: 结果汇总
    desc: 整合所有输出，生成最终汇总结果
    integrations: 
    """
    ctx = runtime.context
    
    # 构建汇总结果
    final_result = {
        "source_video": {
            "title": state.video_title,
            "url": state.video_url
        },
        "products": {
            "short_video": state.short_video_url,
            "learning_guide": state.learning_guide,
            "podcast_audio": state.podcast_audio_url
        }
    }
    
    return ResultSummaryOutput(
        final_result=final_result
    )

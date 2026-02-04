from typing import Optional
from pydantic import BaseModel, Field


class GlobalState(BaseModel):
    """全局状态定义"""
    domain: str = Field(..., description="领域关键词（如：科技、教育、娱乐）")
    video_title: str = Field(default="", description="视频标题")
    video_description: str = Field(default="", description="视频描述")
    video_url: str = Field(default="", description="视频URL")
    transcript: str = Field(default="", description="视频转录文本")
    short_video_url: str = Field(default="", description="生成的短视频URL")
    learning_guide: str = Field(default="", description="学习指南Markdown内容")
    podcast_audio_url: str = Field(default="", description="播客音频URL")
    final_result: dict = Field(default={}, description="最终汇总结果")


class GraphInput(BaseModel):
    """工作流的输入"""
    domain: str = Field(..., description="领域关键词（如：科技、教育、娱乐）")


class GraphOutput(BaseModel):
    """工作流的输出"""
    final_result: dict = Field(..., description="最终汇总结果，包含所有产物")


class HotspotCaptureInput(BaseModel):
    """热点捕获节点的输入"""
    domain: str = Field(..., description="领域关键词")


class HotspotCaptureOutput(BaseModel):
    """热点捕获节点的输出"""
    video_title: str = Field(..., description="视频标题")
    video_description: str = Field(..., description="视频描述")
    video_url: str = Field(..., description="视频URL")
    transcript: str = Field(..., description="视频转录文本")


class VideoRecreationInput(BaseModel):
    """AI视频二创节点的输入"""
    transcript: str = Field(..., description="视频转录文本")
    video_title: str = Field(..., description="原始视频标题")


class VideoRecreationOutput(BaseModel):
    """AI视频二创节点的输出"""
    short_video_url: str = Field(..., description="生成的短视频URL")


class LearningGuideInput(BaseModel):
    """深度学习指南节点的输入"""
    transcript: str = Field(..., description="视频转录文本")
    video_title: str = Field(..., description="原始视频标题")


class LearningGuideOutput(BaseModel):
    """深度学习指南节点的输出"""
    learning_guide: str = Field(..., description="学习指南Markdown内容")


class PodcastScriptInput(BaseModel):
    """播客对谈脚本节点的输入"""
    transcript: str = Field(..., description="视频转录文本")
    video_title: str = Field(..., description="原始视频标题")


class PodcastScriptOutput(BaseModel):
    """播客对谈脚本节点的输出"""
    podcast_audio_url: str = Field(..., description="播客音频URL")


class ResultSummaryInput(BaseModel):
    """结果汇总节点的输入"""
    video_title: str = Field(..., description="视频标题")
    video_url: str = Field(..., description="视频URL")
    short_video_url: str = Field(default="", description="短视频URL")
    learning_guide: str = Field(default="", description="学习指南内容")
    podcast_audio_url: str = Field(default="", description="播客音频URL")


class ResultSummaryOutput(BaseModel):
    """结果汇总节点的输出"""
    final_result: dict = Field(..., description="最终汇总结果")

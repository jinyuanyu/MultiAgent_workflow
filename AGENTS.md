## 项目概述
- **名称**: 趋势内容多维转换专家
- **功能**: 自动追踪YouTube热点，将热门视频深度转化为剪辑短片、学习指南和播客节目

### 节点清单
| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| hotspot_capture | `nodes/hotspot_capture_node.py` | task | 搜索YouTube热门视频并提取信息 | - | - |
| video_recreation | `nodes/video_recreation_node.py` | agent | 分析高光时刻并生成短视频 | - | `config/video_recreation_cfg.json` |
| learning_guide | `nodes/learning_guide_node.py` | agent | 生成结构化学习指南 | - | `config/learning_guide_cfg.json` |
| podcast_script | `nodes/podcast_script_node.py` | agent | 改编对谈脚本并语音合成 | - | `config/podcast_script_cfg.json` |
| result_summary | `nodes/result_summary_node.py` | task | 整合所有输出 | - | - |

**类型说明**: task(task节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## 子图清单
| 子图名 | 文件位置 | 功能描述 | 被调用节点 |
|-------|---------|------|---------|-----------|
| - | - | - | - |

## 集成使用
- 节点 `hotspot_capture` 使用联网搜索集成
- 节点 `video_recreation` 使用大语言模型集成、视频生成大模型集成
- 节点 `learning_guide` 使用大语言模型集成
- 节点 `podcast_script` 使用大语言模型集成、语音大模型集成

## 工作流架构
1. **热点捕获阶段**: 输入领域关键词，搜索YouTube热门视频，提取视频信息
2. **并行处理阶段**: 三个任务并行执行
   - AI视频二创：分析高光时刻 → 生成60秒短视频
   - 深度学习指南：生成Markdown格式学习指南
   - 播客对谈脚本：改编为双人对谈 → 语音合成
3. **结果汇总阶段**: 整合所有输出，返回统一结果

## 输入输出

**输入参数**:
- `domain`: 领域关键词（如：科技、教育、娱乐）

**输出结果**:
- `source_video`: 原始视频信息（标题、URL）
- `products`: 转换产物
  - `short_video`: 短视频URL
  - `learning_guide`: Markdown学习指南
  - `podcast_audio`: 播客音频URL

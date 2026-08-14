# v04 严格同步短视界与反向确认

## 目标

解决视频中约以 chunk 为周期的大幅往返：机器人执行一段轨迹后，新推理再次生成相似动作片段，新 chunk 开头又把机器人带回前一姿态。

## 对现象的最终判断

严格同步时，不能简单归因于“异步旧状态”。更准确的机制是 chunk 重启不连续：

```text
chunk 1: A -> B -> C -> D
robot reaches near D
chunk 2: A' -> B' -> C' -> D'
robot is pulled from D toward A', then moves toward D' again
```

该模型请求中 `history_action=None`，模型没有显式获得上一段已执行 action。固定 diffusion seed 还可能使相近视觉/状态输入稳定地产生相似轨迹模式。以上是代码和现象支持的工作假设，不是仅凭视频即可证明的唯一原因。

## 相对 v03 的具体修改

### 严格同步短视界

默认 `--sync_execute_steps 4`：

1. 发送一份完整最新观测。
2. 等待唯一一个推理请求返回。
3. 从实际关节反馈初始化控制目标。
4. 对返回 chunk 平滑和边界衔接。
5. 只执行前 4 步。
6. 执行完后读取后台维护的最新完整 `obs_horizon`。
7. 再发送下一次请求并等待。

没有 RTC，也没有多个在途请求。

若服务端返回 16 步，30 Hz 下：

```text
完整 chunk 开环时长 = 16 / 30 ~= 0.533 s
当前执行时长         =  4 / 30 ~= 0.133 s
```

短视界将一次承诺给预测轨迹的时间缩短约四分之三。

### 当前反馈重新对齐

每次收到新 chunk 后，通过最新观测的 `state[-1, :14]` 调用 `reset_arm_target()`，使平滑起点基于实际关节反馈，而不是上一段计划值。

### 反向确认

默认：

```text
reverse_confirm_steps = 3
reverse_threshold_rad = 0.004
```

对每个关节独立判断：

- 误差小于阈值时不触发反向保护。
- 当前目标方向与已确认方向相反时开始计数。
- 连续 3 个控制点都要求反向后才允许掉头。
- 单次或短暂相反预测被置零。

这是 debounce/hysteresis 风格的工程保护，不是 ACT 原论文 temporal ensemble 的等价实现。

### 保留的平滑层

- chunk 内 5 点平滑。
- chunk 开头 3 步边界混合。
- 在线低通 `alpha=0.25`。
- 单步限幅 `0.02 rad`。
- 30 Hz 模型 action 与 30 Hz 命令下发。

## 理论依据

### Receding-horizon / MPC 思想

短视界执行与 receding-horizon control 的思想一致：只执行当前计划的一小段，然后利用新状态重新规划。它降低模型预测误差和环境变化在长开环段中的累积。

### Action chunking 与 temporal consistency

ACT 通过一次预测多步 action 降低有效决策时域，但论文同时讨论 temporal aggregation，以融合不同时间预测的重叠 action。当前实现没有完整 temporal ensemble，因此用短视界和边界衔接降低跨 chunk 不连续。

### 反向确认

反向确认类似输入 debounce：要求方向变化持续存在才接受，过滤预测噪声引起的快速符号翻转。它减少往返触发频率，但会延迟真实反向动作。

## 为什么预期能 work

- 不再执行容易与下一次预测冲突的 chunk 后 12 步，直接减小一次错误方向的最大位移。
- 每次重新规划前都已完成短视界执行，观测包含最新机器人和衣物状态。
- 真实反馈用于建立新 chunk 的起点，降低首帧位置不连续。
- 偶发反向点不会立即改变运动方向。

## 为什么不能保证彻底根治

- 如果模型连续多次稳定要求反方向，反向确认最终会放行；这是必要的，否则机器人无法完成真正需要反向的折衣动作。
- 若模型观测不能区分任务阶段，短视界可能反复执行同一段动作的前几步。
- `history_action=None` 仍未改变，服务端没有显式输入已执行动作历史。
- 当前没有实现“在新 chunk 中搜索最接近当前姿态的点”或完整 temporal action ensemble。
- 推理较慢时，严格同步会在短视界之间保持最后位置，可能出现停顿，但不会继续盲目执行长 chunk。

## 推荐验证和判定方法

每次请求记录：

```text
q_request   发送请求时的实际 14 维关节
q_receive   收到 chunk 时的实际关节
a_first     新 chunk 第一条 action
a_exec_end  实际执行的第 4 条 action
q_feedback  每周期真实反馈
```

重点指标：

- `max(abs(a_first - q_receive))`：chunk 首帧不连续。
- 新旧 chunk 主方向余弦相似度：判断是否周期性反向。
- 单位时间反向次数。
- 实际关节摆动峰峰值。
- 严格同步停顿时长和任务完成率。

调参顺序：

1. 往返仍明显：`--sync_execute_steps 4 -> 2`。
2. 偶发掉头：`--reverse_confirm_steps 3 -> 4`。
3. 动作太慢：先提高 `arm_filter_alpha`，不要先增加短视界长度。
4. 真正需要快速反向的动作被压制：降低确认步数或提高反向阈值。

## 安全边界

- `reverse_confirm_steps` 是现场保护，不应代替模型修复。
- 任何参数变化都应先在低速、空载、可急停环境测试。
- 关节速度、加速度和碰撞约束最终应由 Galbot SDK/底层控制器保证；客户端位置限幅不是功能安全层。

## 后续更标准的方案

- 实现 ACT temporal action ensemble，融合多个历史预测在同一执行时刻的 action。
- 把已执行 action 作为 `history_action` 输入模型，前提是 checkpoint 支持并按该方式训练。
- 新 chunk 中搜索与当前姿态最近且时间方向合理的 action 索引。
- 使用带速度/加速度/jerk 约束的轨迹生成器，而不是位置点低通。

## 参考资料

- ACT 原论文和 ALOHA 项目页。
- Model Predictive Control 参考教材。
- Feedback Systems 反馈、时延和稳定性章节。
- Real-Time Execution of Action Chunking Flow Policies，可作为未来异步平滑执行的参考；当前版本没有采用 RTC。

完整链接见 `docs/REFERENCES.md`。

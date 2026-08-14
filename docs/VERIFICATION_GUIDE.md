# 验证指南

## 目标

把“感觉更快/更稳”转化为可比较指标，避免只依赖单次视频观察。

## 通信验证

每次请求记录：

- JPEG 编码前图像尺寸。
- JPEG quality。
- MessagePack 请求总字节数。
- pack 时间。
- WebSocket send/round-trip 时间。
- 服务端 decode、policy 和 debug I/O 时间。

建议使用固定的 100 组观测，对 raw RGB 与 JPEG quality 70/75/85 分别测 P50/P95/P99。

验收建议：

- 图片数严格等于 `4 x obs_horizon`。
- state shape 严格等于 `(obs_horizon, 16)`。
- JPEG 路径请求字节数显著小于 raw 路径。
- action 与 raw 路径对比不出现任务不可接受的系统偏差。

## 控制周期验证

记录每次 `set_joint_commands()` 的 monotonic timestamp：

```text
dt[k] = t[k] - t[k-1]
```

计算 P50/P95/P99、最大间隔和超出 `1.5 / hz` 的周期数量。

## Chunk 内抖动验证

对 14 个手臂关节分别记录：

```text
raw_action
smoothed_plan
sent_target
joint_feedback
```

指标：

- 相邻点最大变化。
- RMS 速度。
- RMS 加速度。
- RMS jerk。
- 高频能量比例。

不能只优化平滑指标，还要同时检查任务耗时、接触时机和成功率。

## 跨 Chunk 往返验证

每轮记录：

```text
request_id
q_request
q_receive
action[0]
action[execute_steps - 1]
feedback trajectory
```

检查：

- 新 chunk 第一帧与当前反馈的距离。
- 相邻 chunk 主运动方向是否反向。
- 每分钟方向反转次数。
- 关节和末端位置峰峰值。

## A/B 顺序

1. v00：确认原始基线。
2. v01：只判断通信收益。
3. v02：只判断控制周期长暂停是否消失。
4. v03：判断 chunk 内小幅抖动是否下降。
5. v04：判断 chunk 周期大幅往返是否下降。

每次只改变一个版本或一个参数，使用相同衣物初始布局、相机位置和机器人起始姿态。

## 安全

- 首次测试使用低速度、空载或软物体，并保持急停可用。
- 客户端限幅不是底层安全约束。
- 任何关节异常、控制周期长时间中断或 SDK 返回失败都应立即停止测试。

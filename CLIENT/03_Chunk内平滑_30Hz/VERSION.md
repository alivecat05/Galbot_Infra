# v03 Chunk 内平滑与 30 Hz 限幅

## 目标

在 v02 已消除大块控制停顿后，进一步抑制 action chunk 内相邻预测点的高频左右变化和位置阶跃。

## 相对 v02 的具体修改

### 5 点对称加权平滑

新增 `smooth_action_plan()`，默认 `--chunk_smoothing_window 5`。

对 14 个手臂关节应用对称三角权重，边界使用 edge padding。夹爪两维保持模型原值。

形式为：

```text
q_smooth[k] = sum_i w[i] q_model[k+i]
sum_i w[i] = 1
```

该操作衰减逐点正负交替等高频分量。

### 在线一阶低通

默认 `--arm_filter_alpha 0.25`：

```text
q_filtered[k] = q_sent[k-1] + alpha (q_desired[k] - q_sent[k-1])
```

`alpha` 越小越平滑，同时跟踪延迟越大。

### 单周期变化限幅

默认 `--max_arm_step_rad 0.02`：

```text
delta = clip(q_desired - q_previous, -0.02, 0.02)
```

在 30 Hz 下，对单关节给出的离散命令变化上限对应约 `0.6 rad/s` 的命令斜率上限。它不是机器人真实速度保证，因为底层控制器、负载和反馈仍会影响实际运动。

### 30 Hz 运行

- `--hz 30`。
- `--command_hz 30`。
- 每个模型 action 点默认下发一次，不额外使用 90 Hz 重复位置命令。

### 本版执行语义

- 保留后台观测和边界混合。
- 保留推理与完整 chunk 执行重叠。
- 执行完整返回 chunk。
- 不启用跨 chunk 反向确认。

## 理论依据

### 低通滤波

机器人抖动常表现为 action 序列中的高频成分。移动平均和一阶指数平滑都是离散低通滤波器，能降低 Nyquist 附近快速交替的命令成分。

### 斜率限制

即便滤波后的单个异常点仍较大，slew-rate limiter 也能限制一个周期内的最大目标变化，避免离散位置命令隐含过大的速度要求。

### 为什么不使用死区

默认 `--arm_deadband_rad 0`。死区会把小变化暂时吞掉，误差累积到阈值后一次释放，可能产生 stick-slip 式的小跳动。连续低通比硬死区更适合该现象。

## 为什么预期能 work

- 若抖动来源是 chunk 内相邻 action 点的正负交替，低通会直接降低其幅度。
- 双层滤波分工不同：整段平滑利用未来 action 点，在线低通保证实际下发连续。
- 限幅提供确定的最坏单步变化边界，不依赖模型输出分布。

## 风险和边界

- 所有低通都会引入相位滞后。过强可能让机器人错过接触时机或无法及时响应衣物移动。
- 对接触丰富的折衣任务，固定 `0.02 rad` 不一定适合所有关节；肩部和腕部可能需要不同限制。
- 该版本只能减小摆动幅度。如果模型每次重新生成从 A 到 D 的相似完整片段，执行完整 chunk 仍可能产生 A-D-A-D 往返。
- 对称平滑使用未来 action，只适用于模型一次返回完整 chunk 的场景。

## 推荐验证

- 同时记录 raw action、plan-smoothed action、最终 sent target 和真实 feedback。
- 比较 `max(abs(diff()))`、RMS jerk 和任务成功率。
- 参数扫描建议：`alpha={0.2,0.25,0.4,0.6}`，`max_step={0.01,0.02,0.04}`。
- 不能只看“抖动最小”，还要验证到达时间、夹取成功率和布料状态变化。

## 参考资料

- Oppenheim/Schafer 离散时间信号处理中的 FIR 与低通概念。
- Åström/Murray 反馈控制中的相位滞后和稳定性。
- ACT action chunk 与 temporal aggregation 的原始工作。

完整链接见 `docs/REFERENCES.md`。

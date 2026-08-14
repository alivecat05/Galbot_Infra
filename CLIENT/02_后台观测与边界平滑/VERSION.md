# v02 后台观测与 chunk 边界平滑

## 目标

解决控制线程在采集完整 `obs_horizon` 时停止下发关节命令的问题，并降低新旧 action chunk 首尾不连续造成的瞬时跳变。

## 相对 v01 的具体修改

### 后台观测历史

新增 `ObservationHistoryWorker`：

- 以 `1 / hz` 为目标周期采集四路图片和 16 维状态。
- 在后台完成缩放和 JPEG 编码。
- 通过锁维护最近 `obs_horizon` 条观测。
- `snapshot()` 返回最新完整模型输入。
- 控制线程不再调用连续阻塞式历史采集。

发送给模型的内容仍是：

```text
images: 4 x obs_horizon
state:  obs_horizon x 16
```

### chunk 边界衔接

新增 `blend_plan_start()`：

- 获取当前真实 14 维手臂关节状态。
- 将新 chunk 前 3 个 action 从当前姿态渐变到模型轨迹。
- 只平滑手臂 14 维，不改变两个夹爪目标。

近似公式：

```text
q_cmd[k] = (1 - alpha_k) q_current + alpha_k q_model[k]
alpha_k 从 0 增长到 1
```

### 本版执行语义

- 后台持续更新观测。
- 在当前 chunk 开始后发送下一次请求，使推理和当前 chunk 执行重叠。
- 仍执行完整 chunk。
- 本版不是最终严格同步版，保留它是为了观察“去掉采集阻塞”和“边界衔接”的独立效果。

## 理论依据

### 控制周期抖动

数字控制器依赖相对稳定的采样/执行周期。若实际间隔 `dt_k` 大幅变化，相同的位置序列会对应不同速度和加速度：

```text
velocity[k]     ~= (q[k] - q[k-1]) / dt_k
acceleration[k] ~= (velocity[k] - velocity[k-1]) / dt_k
```

后台采集把图像 I/O 和 JPEG 编码移出关节命令关键路径，减少 `dt_k` 的长尾。

### 边界连续性

如果上一 chunk 末尾为 `q_old_end`，新 chunk 第一帧为 `q_new_0`，直接切换会产生位置跳变：

```text
delta_boundary = q_new_0 - q_old_end
```

线性衔接将该变化分摊到多个周期，降低等效速度和加速度冲击。

## 为什么预期能 work

- 图片采集与控制下发是相互独立的任务，线程隔离可减少阻塞传播。
- 双缓冲式 snapshot 让推理请求读取一份完整、一致的历史，不需要减少 `obs_horizon`。
- 边界混合只修改 action chunk 的开头，不改变后续模型轨迹目标。

## 风险和边界

- Galbot SDK 是否允许多个线程同时调用相机、状态和关节接口，需要在目标设备确认线程安全。
- Python GIL 不会阻止 OpenCV/C 扩展并行，但纯 Python 或 SDK 内部锁仍可能竞争。
- 线性位置插值不是严格的速度/加速度受限轨迹生成。
- 如果新 chunk 本身整体方向错误，边界平滑只会让错误变化更缓慢，不能消除持续往返。
- 本版在 chunk 开始时预取下一次推理，观测与真正使用 action 的时刻仍可能存在偏移。

## 推荐验证

- 记录 `set_joint_commands()` 调用间隔的 P50/P95/P99。
- 对比后台线程启用前后的最大控制空窗。
- 记录每个 chunk 边界的 `max(abs(q_new_0 - q_feedback))`。
- 若 SDK 出现线程冲突，改为单独传感器进程或由 SDK 官方流式接口提供缓存。

## 参考资料

- Åström 与 Murray《Feedback Systems》中的采样、反馈和时延讨论。
- Python `threading` 和 `queue` 官方文档。
- ACT 对多视角图像、关节状态和 action chunk 的定义。

完整链接见 `docs/REFERENCES.md`。

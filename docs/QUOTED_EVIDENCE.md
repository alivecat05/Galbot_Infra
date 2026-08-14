# 原文证据索引

本文件集中列出教程中使用的原始资料短引文。每条引文都包含中文解释、与代码的关系，以及引用边界。

访问日期：2026-08-14。

## 1. MessagePack 可以直接携带二进制字节

来源：MessagePack specification, Type system  
https://github.com/msgpack/msgpack/blob/master/spec.md#type-system

> “Binary ... represents a byte array”

中文：MessagePack 的 Binary 类型表示一个字节数组。

它支持的结论：JPEG 编码结果本来就是 bytes，可以直接使用 MessagePack binary 发送，不需要先转成字符串。

对应代码：

```python
"data": payload.tobytes()
```

它不能证明：所有 WebSocket 库都会自动采用 MessagePack binary。这个项目成立，是因为当前 `WebsocketClientPolicy` 的协议使用 MessagePack，并能序列化 Python `bytes`。

## 2. Base64 为什么约增加三分之一

来源：RFC 4648, Section 4  
https://www.rfc-editor.org/rfc/rfc4648.html#section-4

> “24-bit groups of input bits as output strings of 4 encoded characters.”

中文：每 24 bit，也就是 3 个 8-bit 字节，会变成 4 个编码字符。

因此完整编码组的比值为：

```text
输入 = 3 bytes
输出 = 4 characters
4 / 3 = 1.333...
```

它支持的结论：在不考虑尾部 padding 和字符串对象额外开销时，Base64 比原始二进制约大 33.3%。

对应方案：JPEG bytes 直接进入 MessagePack Binary，不走 Base64。

它不能证明：Base64 永远不可用。若协议只能传文本，Base64 可能是兼容方案；当前协议原生支持 binary，所以没有必要。

## 3. `np.savez_compressed` 确实会生成压缩归档

来源：NumPy official documentation  
https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html

> “Save several arrays into a single file in compressed .npz format.”

以及 Notes 中：

> “The .npz file format is a zipped archive”

中文：该函数把多个数组写入一个压缩 `.npz` 文件；`.npz` 是 ZIP 归档。

它支持的结论：该调用既包含数组归档和压缩，也包含文件写入。若在响应返回前同步执行，它属于请求关键路径。

对应修改：服务端 `--debug_dir` 默认从固定目录改为 `None`。

它不能证明：这一步一定耗时几秒。实际耗时取决于图片数量、磁盘、CPU 和数据可压缩性，必须通过计时验证。

## 4. 为什么后台线程适合相机和网络类等待任务

来源：Python `threading` official documentation  
https://docs.python.org/3/library/threading.html

> “Threads are particularly useful when tasks are I/O bound”

官方文档还说明，线程在同一进程中并发运行并共享内存空间。

中文：线程特别适合很多时间花在等待外部资源的 I/O-bound 任务，例如文件和网络操作。

它支持的结论：把相机读取、等待和编码流水线放到后台线程，可以避免控制主循环主动等待整段历史采集。

对应代码：`ObservationHistoryWorker`。

它不能证明：线程一定提高 JPEG 编码速度，也不能证明 Galbot SDK 的所有接口线程安全。SDK 线程安全和锁竞争仍要在设备上验证。

## 5. ACT 为什么使用 action chunk

来源：Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware  
https://arxiv.org/html/2304.13705

> “errors in the policy can compound over time”

中文：策略误差会随着时间累积。

论文在 Action Chunking and Temporal Ensemble 一节说明，固定 chunk 大小为 `k`，每 `k` 步接收观测、生成接下来的 `k` 个动作并顺序执行，从而降低有效任务时域。

它支持的结论：一次预测多个未来 action 是 ACT 设计的一部分，不能简单把模型改成只输出一个点，也不能通过删除 `obs_horizon` 来处理延迟。

对应代码：服务端仍返回完整 action chunk，客户端根据版本决定执行完整 chunk 或短前缀。

## 6. 朴素 action chunk 为什么可能抖

来源：同一 ACT 论文，Section IV-A  
https://arxiv.org/html/2304.13705#S4.SS1

> “A naïve implementation of action chunking can be sub-optimal”

论文紧接着指出：

> “can result in jerky robot motion.”

中文：朴素 action chunk 实现可能不是最优；每隔 `k` 步突然纳入新观测，可能导致机器人动作不顺滑。

它支持的结论：chunk 之间的离散切换本身就是已知问题，不能只盯着网络延迟或电机机械抖动。

对应修改：

- v02 的 chunk 边界混合。
- v04 的短视界重新规划。

它不能证明：当前视频的唯一原因就是论文描述的问题。现场还可能叠加模型阶段判断、底层位置控制和接口时序问题。

## 7. ACT temporal ensemble 与普通低通不同

来源：同一 ACT 论文，Section IV-A  
https://arxiv.org/html/2304.13705#S4.SS1

> “we aggregate actions predicted for the same timestep.”

中文：论文融合的是针对同一个执行时刻做出的多次预测。

这与 v03 的普通邻点平滑不同：

```text
v03 邻点平滑:
同一个 chunk 内，把 t-2, t-1, t, t+1, t+2 的目标做平滑

ACT temporal ensemble:
来自不同推理时刻的多个 chunk，都对执行时刻 t 有预测，再融合这些预测
```

它支持的结论：v03 可以抑制 chunk 内高频波动，但不等于已经实现 ACT temporal ensemble。

对应当前状态：v04 使用短视界、边界衔接和反向确认作为部署保护，完整 temporal ensemble 仍属于后续方案。

## 8. ACT 的 action 是目标关节位置

来源：同一 ACT 论文  
https://arxiv.org/html/2304.13705

> “An action here corresponds to the target joint positions for both arms”

中文：论文中的 action 对应双臂目标关节位置。

它支持的结论：新 chunk 第一条 action 若与当前真实关节位置不连续，会直接表现为位置目标跳变；这也是边界衔接和当前反馈重新对齐有意义的原因。

它不能自动证明本项目 checkpoint 与 ACT 原实现完全相同。当前项目应通过 `ACTION_KEYS`、归一化统计和模型输出定义确认 action 语义。

## 9. 反馈控制资料提供什么依据

来源：Åström and Murray, Feedback Systems  
https://fbswiki.org/wiki/index.php/Feedback_Systems:_An_Introduction_for_Scientists_and_Engineers

官方目录明确覆盖：

> “Stability”

以及：

> “Control Using Estimated State”

这些内容支持对反馈、状态、稳定性和时序影响的基础解释。

引用边界：教材提供通用控制理论，但没有直接证明当前 Galbot 的摆动由某个单一参数引起。具体原因必须结合命令、反馈和时间戳测量。

## 10. 引用使用原则

- 引文只用于支持资料明确陈述的内容。
- 引文之后必须区分“资料证明”“基于资料的推导”和“现场工作假设”。
- `reverse_confirm_steps` 是工程启发式，不包装成 ACT 论文原算法。
- v04 借鉴滚动重规划思想，但不是完整 MPC。
- OpenCV `imencode/imdecode` 的 API 用途在 `REFERENCES.md` 给出官方链接；由于官方站点本次触发安全验证，本文件不放未经本次逐字复核的长引文。

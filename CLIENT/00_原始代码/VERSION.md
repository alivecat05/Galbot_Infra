# v00 原始代码基线

## 版本性质

这是用户最初提供的客户端和服务端精确快照。该版本用于建立性能和控制行为基线，不包含本轮优化。

## 原始通信路径

客户端从四路相机获取 SDK 压缩帧。当服务端 metadata 没有声明 `encoded_images=true` 时，客户端调用 `decode_rgb()`，把每张图解码成 `H x W x 3` 的 `uint8` NumPy 数组，再交给 WebSocket/MessagePack 序列化。

一次请求包含：

```text
图片数 = 4 x obs_horizon
原始 RGB 字节数 = 4 x obs_horizon x H x W x 3
状态数 = obs_horizon x 16 x sizeof(float32)
```

以 `640 x 480` 为例，单张 RGB 图约为 `0.88 MiB`。若 `obs_horizon=4`，仅图片理论值约为 `14.1 MiB`，还未包含 MessagePack 容器和对象创建开销。

## 原始服务端路径

- 只接受可转换为 NumPy RGB 数组的图片。
- 每张图在服务端转成 PIL，再缩放到 `224 x 224`。
- `debug_dir` 默认启用。
- 每次响应前同步执行 `np.savez_compressed()`，保存所有图片、状态和 action。
- 固定 diffusion seed，使相同或相近输入更容易得到可重复输出。
- `history_action=None`，模型没有显式接收上一段已执行 action。

## 原始同步控制路径

虽然函数名是 `run_sync()`，内部仍使用 `AsyncPolicyWorker`：

1. 等待当前 chunk。
2. 逐点执行 action。
3. 在 chunk 中点调用 `capture_observation_history()`。
4. 该函数在控制线程内连续采集整个 `obs_horizon`，期间包含 `sleep(1 / hz)`。
5. 采集完成后提前发送下一次推理请求。

因此它属于“单 WebSocket、单请求 worker、chunk 中途预取”，不是严格的“执行结束后再采集、再推理”。

## 已知问题和理论解释

### 通信开销

未经压缩的 RGB 数组具有很高的空间冗余。网络时间近似为：

```text
T_network = payload_bytes / effective_bandwidth + RTT
```

减少 payload 是降低传输时间最直接的方法，同时也减少 MessagePack 打包、对象遍历、内存复制和服务端反序列化成本。

### 同步调试 I/O

`np.savez_compressed()` 包含压缩计算和磁盘写入，并位于返回路径上：

```text
T_response = T_decode + T_policy + T_compress_debug + T_disk + T_send
```

任何磁盘抖动都会直接增加客户端等待时间。

### 控制停顿

在 action 执行线程中阻塞采集整个历史，会造成关节指令时间间隔不均匀。位置命令的实际采样间隔由固定的 `1/hz` 变成“正常周期与长暂停交替”，容易出现走停感和 chunk 周期抖动。

## 基线验证

- 记录每次请求的序列化字节数。
- 记录 `capture_observation_history()` 的总耗时。
- 记录相邻 `set_joint_commands()` 调用时间差。
- 记录服务端 `np.savez_compressed()` 耗时。
- 记录每个 chunk 的 `max(abs(diff(action[:, :14])))`。

## 参考资料

见仓库 `docs/REFERENCES.md` 中 MessagePack、JPEG、反馈控制和 ACT 相关条目。

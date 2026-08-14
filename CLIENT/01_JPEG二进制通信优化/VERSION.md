# v01 JPEG 二进制通信优化

## 这一版为什么出现

v00 最明显的问题是一次请求可能有十几 MB。模型最终只看 `224 x 224`，却先把高清 RGB 全部传到服务器再缩小，浪费了带宽、内存复制和序列化时间。

这一版只做一件事：把“运输的箱子”变小。它故意保留 v00 的阻塞采集和动作执行方式，方便证明延迟下降确实来自通信，而不是同时改了控制流程。

最直观的类比：

```text
v00: 邮寄展开后的高清海报
v01: 在现场缩成证件照并压缩后邮寄
```

如果 v01 通信明显变快但机器人仍走停或抖动，这是预期现象，因为控制问题要到 v02/v03/v04 才处理。

## 目标

在不改变 `obs_horizon`、相机数量、图片顺序和状态维度的前提下，降低一次推理请求的网络载荷和序列化成本。

## 相对 v00 的具体修改

### 客户端

- 新增 `encode_transport_image()`。
- SDK 相机帧先在机器人侧解码。
- 使用 `cv2.resize(..., INTER_AREA)` 缩放到 `224 x 224`。
- 使用 `cv2.imencode('.jpg', ..., JPEG_QUALITY=75)` 编码。
- 发送结构改为：

```python
{
    "__encoded_image__": True,
    "codec": "jpeg",
    "width": 224,
    "height": 224,
    "data": jpeg_bytes,
}
```

- `data` 是 MessagePack 的 binary/bytes 类型，不是 Base64 字符串。
- 新增 `--transport_image_size`、`--transport_jpeg_quality` 和 `--raw_image_transport`。
- 从服务端 metadata 读取 `transport_image_size`，避免客户端和服务端尺寸配置不一致。

### 服务端

- 新增 `decode_wire_image()`，接受 JPEG/PNG bytes。
- 保留原始 RGB NumPy 数组兼容路径。
- metadata 增加：

```python
"encoded_images": True
"transport_image_size": [224, 224]
```

- 图片已经是 `224 x 224` 时跳过重复缩放。
- `debug_dir` 默认改为 `None`，不再默认同步压缩落盘。
- 增加可选 CUDA dtype：`auto/fp32/fp16/bf16`。

### 本版刻意没有解决的内容

- 保留阻塞式历史采集和 chunk 中点预取，用于隔离观察通信优化的效果。
- 不启用 chunk 内低通、限幅、短视界和反向确认。

## 理论依据

### 为什么二进制 JPEG 比原始 RGB 小

原始 RGB 每像素固定 3 字节。JPEG 利用颜色变换、空间相关性、DCT 量化和熵编码压缩自然图像。模型最终只使用 `224 x 224`，因此先缩放再传输避免发送模型不会使用的高分辨率像素。

理论载荷对比：

```text
raw_224 = 224 x 224 x 3 = 150,528 bytes/image
jpeg_224_q75 ~= 8,000 to 30,000 bytes/image  # 依场景而变
```

单张通常减少约 5 到 18 倍；若原始输入是 `640 x 480`，先缩放再 JPEG 的总下降可更高。

### 为什么不使用 Base64

Base64 每 3 字节变成 4 个 ASCII 字符，理想情况下体积膨胀约：

```text
4 / 3 - 1 = 33.3%
```

MessagePack 原生支持 binary 类型，因此 Base64 不增加兼容价值，只会增加数据量和编解码 CPU。

### 为什么关闭同步 debug I/O

响应必须等同步压缩和磁盘写入结束才能发送。默认关闭后，关键路径缩短为图片解码、模型推理和响应序列化。

## 为什么预期能 work

- 模型原本在服务端也会把图片缩放到 `224 x 224`；现在只是把该缩放提前到客户端。
- 服务端仍恢复成 RGB 图并按照原来的 PIL/model preprocessing 输入模型。
- 图片数量、时间维、相机顺序和 state 不变，因此模型结构契约没有变化。
- JPEG 有损误差通常远小于大幅缩放带来的信息变化，但必须用任务数据验证。

## 风险和边界

- JPEG 质量过低会破坏衣物边缘、纹理和夹爪接触区域。
- 如果训练预处理要求先裁剪再缩放，客户端直接缩放可能改变分布。
- `fp16/bf16` 只影响推理性能，不保证所有 GPU 和算子都支持。
- 该版本只解决通信与服务端 I/O，不解决控制抖动。

## 推荐验证

- 比较同一观测的 raw RGB 与 JPEG 请求字节数。
- 比较客户端 pack/send 时间和端到端 RTT。
- 对同一组离线图片分别走原始预处理与 JPEG 路径，比较模型 action 的 MAE/max error。
- JPEG quality 建议从 75 开始，必要时测试 70、80、90。

## 参考资料

- MessagePack specification。
- RFC 4648 Base64。
- OpenCV `imencode`/`imdecode` 文档。
- JPEG ITU-T T.81 标准。

完整链接见 `docs/REFERENCES.md`。

可直接核对的原文、翻译及引用边界见 `docs/QUOTED_EVIDENCE.md` 第 1-3 节。

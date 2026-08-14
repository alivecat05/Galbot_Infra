# Galbot_Infra

Galbot G1 双臂折衣模型的推理服务、部署客户端和版本化优化记录。

## 目录

- `CLIENT/00_原始代码`：用户提供的原始客户端和服务端快照。
- `CLIENT/01_JPEG二进制通信优化`：模型分辨率 JPEG 二进制传输。
- `CLIENT/02_后台观测与边界平滑`：后台观测历史和 chunk 边界衔接。
- `CLIENT/03_Chunk内平滑_30Hz`：30 Hz 下的 chunk 内平滑、低通和限幅。
- `CLIENT/04_严格同步短视界_反向确认_当前版`：严格同步短视界和反向确认。
- `SERVER/inference_server_galbot_fold_clothes.py`：当前服务端入口。
- `docs/REFERENCES.md`：论文、标准和官方文档。
- `docs/VERSION_MATRIX.md`：每一版解决的问题和控制语义对照。
- `docs/VERIFICATION_GUIDE.md`：通信、时序和动作稳定性验证方法。

## 不变约束

- 不修改模型要求的 `obs_horizon`。
- 每次模型输入保持 `4 x obs_horizon` 张图片。
- 图片顺序保持 view-major：四个相机依次排列，每个相机内部从旧到新。
- 状态保持 `(obs_horizon, 16)`。
- 当前实际部署路径使用普通 WebSocket 同步模式，不要求 RTC。

## 版本性质

`v00` 和 `v04` 是精确快照。`v01-v03` 是根据实际改动阶段整理出的可运行里程碑版本，代码行为已经按各阶段重新拆分，不只是不同参数的同一份当前代码。

详细理论和参考资料见各版本目录中的 `VERSION.md`。

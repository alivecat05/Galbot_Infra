# 参考资料

以下资料用于说明本仓库的编码、通信、控制和平滑设计。访问日期：2026-08-14。

## 通信与图像编码

1. MessagePack Specification  
   https://github.com/msgpack/msgpack/blob/master/spec.md  
   依据：MessagePack 原生区分 binary 与 string，可直接承载 JPEG bytes。

2. RFC 4648: Base-N Encodings  
   https://www.rfc-editor.org/rfc/rfc4648  
   依据：Base64 将 24 bit 输入编码为 4 个 6 bit 字符，解释约 33% 的理想体积膨胀。

3. OpenCV Image file reading and writing  
   https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html  
   依据：`cv::imencode`/`cv::imdecode` 的内存编码和解码语义；Python API 对应 `cv2.imencode/imdecode`。

4. OpenCV Geometric Image Transformations  
   https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html  
   依据：缩小图像时 `INTER_AREA` 通常是合适的重采样方法。

5. ITU-T Recommendation T.81 / ISO/IEC 10918-1: JPEG  
   https://www.w3.org/Graphics/JPEG/itu-t81.pdf  
   依据：JPEG 编码的标准定义。

6. NumPy `savez_compressed`  
   https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html  
   依据：该调用创建压缩 ZIP archive；同步位于响应关键路径时会增加 CPU 和磁盘 I/O。

## Python 并发

7. Python `threading`  
   https://docs.python.org/3/library/threading.html  
   依据：后台观测线程、事件和锁的行为。

8. Python `queue`  
   https://docs.python.org/3/library/queue.html  
   依据：线程安全请求/结果队列和 bounded queue。

## 控制与信号处理

9. Karl J. Åström and Richard M. Murray, Feedback Systems  
   https://fbsbook.org/  
   依据：反馈、采样、延迟、稳定性和闭环振荡的基本理论。该书可在线获取。

10. Alan V. Oppenheim and Ronald W. Schafer, Discrete-Time Signal Processing  
    ISBN 978-0131988422  
    依据：FIR、移动平均、频率响应和离散低通滤波。

11. James B. Rawlings, David Q. Mayne, Moritz M. Diehl, Model Predictive Control: Theory, Computation, and Design  
    https://sites.engineering.ucsb.edu/~jbraw/mpc/  
    依据：receding-horizon control，只执行计划的一部分并根据新状态重复优化。

## 机器人模仿学习和 action chunk

12. Tony Z. Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, RSS 2023  
    https://arxiv.org/abs/2304.13705  
    依据：Action Chunking with Transformers，使用 action chunk 降低有效时域，并使用 temporal aggregation 提升执行平滑性。

13. ALOHA / ACT 官方项目页  
    https://tonyzhaozh.github.io/aloha/  
    依据：ACT 的算法说明、多视角图像和关节位置输入，以及 chunk 化关节位置输出。

14. ACT 官方代码库  
    https://github.com/tonyzhaozh/act  
    依据：ACT evaluation、chunk 和 temporal aggregation 的实现参考。

15. Hugging Face LeRobot ACT documentation  
    https://huggingface.co/docs/lerobot/act  
    依据：ACT 的工程实现和配置参考，不替代原论文。

16. Real-Time Execution of Action Chunking Flow Policies  
    https://www.pi.website/download/real_time_chunking.pdf  
    依据：未来如需在推理和动作执行重叠时保持 chunk 连续，可参考其 real-time chunking 思路。当前 v04 明确没有采用 RTC。

## 引用边界

- JPEG、Base64、MessagePack 的结论直接来自标准或官方规范。
- 低通、限幅和短视界的理论来自信号处理及控制教材。
- `reverse_confirm_steps` 是本项目基于现场现象设计的工程启发式，没有声称来自 ACT 论文。
- 当前版本的短视界执行借鉴 receding-horizon 思想，但模型 action 不是在线求解的传统 MPC，因此不应把它描述为完整 MPC 控制器。

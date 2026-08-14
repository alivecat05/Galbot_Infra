# v04 current exact snapshot

Exact current deliverable snapshot:

- 224x224 JPEG MsgPack binary transport.
- Complete `obs_horizon` preserved.
- Background observation history capture.
- Strict synchronous execution; RTC is not required.
- Executes 4 steps from each returned chunk before replanning.
- 5-point within-chunk smoothing and 3-step boundary blending.
- Online low-pass alpha 0.25 and 0.02 rad per-step limit.
- Opposite motion must be confirmed for 3 control steps above 0.004 rad.

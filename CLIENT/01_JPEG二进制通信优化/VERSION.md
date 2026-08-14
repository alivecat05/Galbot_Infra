# v01 JPEG binary transport

Reconstructed milestone profile:

- Complete `4 x obs_horizon` images are preserved.
- Images are resized to 224x224 and JPEG encoded at quality 75.
- JPEG data is carried as MsgPack binary bytes.
- Base64 is not used.
- Later smoothing, limiting, reversal confirmation and short-horizon execution are disabled by default.

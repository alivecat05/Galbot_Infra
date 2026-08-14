# v03 within-chunk smoothing at 30 Hz

Reconstructed milestone profile:

- Includes compressed transport and background observations.
- Uses a 5-point action-plan smoothing window.
- Uses online low-pass alpha 0.25.
- Limits each arm joint to 0.02 rad per model step.
- Executes full chunks and does not require reversal confirmation.

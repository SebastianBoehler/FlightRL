# Proposed title

Improve WiFi streamer throughput with pipelined capture

# Proposed body

## What changed

The streamer previously restarted the camera for every frame and processed
capture, encoding, and transfer sequentially. This PR adds:

- continuous and double-buffered capture, allowing the next frame to be
  captured while the current frame is encoded and transferred
- build options for resolution, capture mode, sensor timing, encoding, and
  profiling
- grouped and verified HM01B0 timing updates
- correct handling of the final partial CPX chunk
- resolution-aware raw frame handling in the OpenCV viewer

Defaults and the raw/JPEG wire formats are unchanged: QVGA, raw, default sensor
timing, and start-stop capture.

Relates to #137.

## Results

Measured on an AI Deck 1.1 while collecting Crazyflie radio telemetry:

| Configuration | Resolution | Rate | Dropped frames |
| --- | --- | ---: | ---: |
| Start-stop JPEG | 324 x 244 | 7.71 FPS | 0 / 480 |
| Pipelined JPEG, 60 FPS timing | 324 x 244 | 17.12 FPS | 0 / 1,020 |
| Pipelined JPEG, 60 FPS timing | 162 x 122 | 55.08 FPS | 0 |

The QVGA path improves by 2.22x without changing the `324x244` JPEG
representation.

## Validation

- Clean builds in `bitcraze/aideck`: default, QVGA pipelined JPEG at 60 FPS,
  and QQVGA pipelined JPEG at 60 FPS.
- OpenCV viewer syntax check.
- Build-time rejection of invalid option values.

# Proposed title

Add configurable packed grayscale streaming

# Proposed body

> Stacked on #157. This branch will be rebased onto upstream `master` after
> that PR merges.

## What changed

This PR adds an optional compact observation format for applications that do
not need full raw or JPEG frames:

- configurable GAP8-side nearest-neighbor downsampling
- packed 4-bit grayscale output, with two pixels per byte
- matching decoding and display support in the OpenCV viewer
- a 65 FPS sensor timing option restricted to QQVGA

Raw and JPEG are unchanged. QVGA/raw/start-stop remains the default.

## Format

The first pixel is stored in the high nibble and the second in the low nibble.
The viewer expands each value to 8-bit grayscale. The measured configuration
uses:

```text
162x122 sensor grayscale
  -> nearest-neighbor 64x48
  -> 4-bit quantization
  -> two pixels per byte
  -> 1,536-byte payload
```

## Results

| Encoding | Output | Payload | Rate | Transport errors |
| --- | --- | ---: | ---: | ---: |
| JPEG | 162 x 122 | ~1,986 bytes | 61.72 FPS | 2 / 3,600 frames |
| Gray4 | 64 x 48 | 1,536 bytes | 64.83 FPS | 0 / 3,840 frames |

The `64x48` output is a build configuration, not a fixed format size.

## Validation

- Clean builds in `bitcraze/aideck`: default and QQVGA/pipelined/65 FPS/gray4
  at `64x48`.
- QVGA with 65 FPS timing is rejected at build configuration.
- Focused odd-pixel nibble-order check and viewer syntax check.

static void randomize_edge_door_camera(
    uint8_t *frame,
    float strength,
    int scene_seed,
    int control_step
) {
    strength = vision_clamp(strength, 0.0f, 1.0f);
    if (strength <= 0.0f) return;
    uint8_t source[SIXDOF_VISION_PIXELS];
    memcpy(source, frame, SIXDOF_VISION_PIXELS);

    /* Optical response stays fixed for an episode. */
    uint32_t profile = vision_hash((uint32_t)scene_seed ^ 0x6a09e667u);
    float gamma = 1.0f + strength * (
        0.85f + 0.30f * vision_unit(profile) - 1.0f
    );
    float contrast = 1.0f + strength * (
        0.88f + 0.24f * vision_unit(profile >> 8) - 1.0f
    );
    float vignette = strength * (
        0.03f + 0.09f * vision_unit(profile >> 4)
    );
    float base_exposure = strength * (
        -4.0f + 8.0f * vision_unit(profile >> 16)
    );

    /* Sensor effects vary per frame but reproduce from scene and step. */
    uint32_t frame_key = vision_hash(
        (uint32_t)scene_seed
            ^ vision_hash((uint32_t)control_step ^ 0xbb67ae85u)
    );
    float exposure = base_exposure + strength * (
        -7.0f + 14.0f * vision_unit(frame_key)
    );
    float phase = 6.283185307f * vision_unit(frame_key >> 8);
    float band_amplitude = strength * (
        1.5f + 2.5f * vision_unit(frame_key >> 16)
    );
    for (int row = 0; row < SIXDOF_VISION_HEIGHT; ++row) {
        int up = row > 0 ? row - 1 : row;
        int down = row + 1 < SIXDOF_VISION_HEIGHT ? row + 1 : row;
        float screen_y = (
            2.0f * ((float)row + 0.5f) / SIXDOF_VISION_HEIGHT - 1.0f
        );
        float band = band_amplitude * sinf(0.63f * (float)row + phase);
        for (int col = 0; col < SIXDOF_VISION_WIDTH; ++col) {
            int left = col > 0 ? col - 1 : col;
            int right = col + 1 < SIXDOF_VISION_WIDTH ? col + 1 : col;
            int pixel = row * SIXDOF_VISION_WIDTH + col;
            float value = (
                4.0f * source[pixel]
                + source[row * SIXDOF_VISION_WIDTH + left]
                + source[row * SIXDOF_VISION_WIDTH + right]
                + source[up * SIXDOF_VISION_WIDTH + col]
                + source[down * SIXDOF_VISION_WIDTH + col]
            ) / 8.0f;
            float normalized = powf(value / 255.0f, gamma);
            float screen_x = (
                2.0f * ((float)col + 0.5f) / SIXDOF_VISION_WIDTH - 1.0f
            );
            float falloff = 1.0f - vignette * (
                screen_x * screen_x + 0.70f * screen_y * screen_y
            );
            uint32_t noise_key = vision_hash(
                frame_key ^ (uint32_t)(pixel + 1) * 0x9e3779b9u
            );
            float noise = strength * (
                vision_unit(noise_key) - 0.5f
            ) * 4.0f;
            float adjusted = vision_clamp(
                127.5f + contrast * (255.0f * normalized - 127.5f)
                    + exposure + band + noise
                    - 255.0f * strength * (1.0f - falloff),
                0.0f,
                255.0f
            );
            frame[pixel] = (uint8_t)(
                17.0f * floorf(adjusted / 17.0f + 0.5f)
            );
        }
    }
}

void flightrl_edge_grounding_from_mask(
    uint8_t *door_mask,
    float *grounding
) {
    int count = 0;
    int min_col = SIXDOF_VISION_WIDTH;
    int max_col = -1;
    int min_row = SIXDOF_VISION_HEIGHT;
    int max_row = -1;
    for (int row = 0; row < SIXDOF_VISION_HEIGHT; ++row) {
        for (int col = 0; col < SIXDOF_VISION_WIDTH; ++col) {
            int pixel = row * SIXDOF_VISION_WIDTH + col;
            if (flightrl_door_airframe_pixel_masked(
                row, col, SIXDOF_VISION_WIDTH, SIXDOF_VISION_HEIGHT
            )) {
                door_mask[pixel] = 0;
            }
            if (!door_mask[pixel]) continue;
            count += 1;
            min_col = min_col < col ? min_col : col;
            max_col = max_col > col ? max_col : col;
            min_row = min_row < row ? min_row : row;
            max_row = max_row > row ? max_row : row;
        }
    }
    int visible = count >= 4;
    grounding[0] = (float)visible;
    grounding[1] = visible
        ? (float)(min_col + max_col) / (SIXDOF_VISION_WIDTH - 1) - 1.0f
        : 0.0f;
    grounding[2] = visible
        ? (float)(min_row + max_row) / (SIXDOF_VISION_HEIGHT - 1) - 1.0f
        : 0.0f;
    grounding[3] = visible
        ? sqrtf(
            (float)((max_col - min_col + 1) * (max_row - min_row + 1))
                / SIXDOF_VISION_PIXELS
        )
        : 0.0f;
}

void flightrl_sixdof_edge_door_observation_scene(
    const float *position,
    const float *quaternion,
    const float *room,
    const float *door,
    const float *obstacle,
    float target_mean,
    int scene_seed,
    float camera_randomization,
    float *door_grounding,
    float *observation
) {
    uint8_t current[SIXDOF_VISION_PIXELS];
    uint8_t door_mask[SIXDOF_VISION_PIXELS];
    render_one(
        position,
        quaternion,
        room,
        obstacle,
        door,
        target_mean,
        scene_seed,
        NULL,
        door_mask,
        current
    );
    randomize_door_camera(current, camera_randomization, scene_seed);
    flightrl_mask_door_airframe(
        current,
        SIXDOF_VISION_WIDTH,
        SIXDOF_VISION_HEIGHT
    );
    flightrl_edge_grounding_from_mask(door_mask, door_grounding);
    for (int pixel = 0; pixel < SIXDOF_VISION_PIXELS; ++pixel) {
        current[pixel] = (uint8_t)(
            17.0f * floorf(current[pixel] / 17.0f + 0.5f)
        );
        observation[pixel] = current[pixel] / 255.0f;
    }
}

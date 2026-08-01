static void configure_coverage(
    const float *position,
    const float *room,
    FlightRLDoorScene *scene
) {
    if (scene->obstacle[0] > 5.0f) {
        scene->coverage[0] = 0.5f * (room[0] + room[1]);
        scene->coverage[1] = 0.5f * (room[2] + room[3]);
        return;
    }
    float obstacle_x = 0.5f * (scene->obstacle[0] + scene->obstacle[1]);
    float obstacle_y = 0.5f * (scene->obstacle[2] + scene->obstacle[3]);
    float dx = obstacle_x - position[0];
    float dy = obstacle_y - position[1];
    float inv = 1.0f / fmaxf(sqrtf(dx * dx + dy * dy), 1.0e-3f);
    float offset = fmaxf(
        0.5f * (scene->obstacle[1] - scene->obstacle[0]),
        0.5f * (scene->obstacle[3] - scene->obstacle[2])
    ) + 0.75f;
    float lateral_x = -dy * inv;
    float lateral_y = dx * inv;
    float candidates[4];
    for (int side = 0; side < 2; ++side) {
        float sign = side == 0 ? -1.0f : 1.0f;
        candidates[2 * side] = clampf(
            obstacle_x + sign * lateral_x * offset,
            room[0] + 0.30f,
            room[1] - 0.30f
        );
        candidates[2 * side + 1] = clampf(
            obstacle_y + sign * lateral_y * offset,
            room[2] + 0.30f,
            room[3] - 0.30f
        );
    }
    float first_clearance = hypotf(
        candidates[0] - obstacle_x,
        candidates[1] - obstacle_y
    );
    float second_clearance = hypotf(
        candidates[2] - obstacle_x,
        candidates[3] - obstacle_y
    );
    int selected = second_clearance > first_clearance ? 1 : 0;
    scene->teacher_side = selected == 0 ? -1.0f : 1.0f;
    scene->coverage[0] = candidates[2 * selected];
    scene->coverage[1] = candidates[2 * selected + 1];
}

static void disable_obstacle(FlightRLDoorScene *scene) {
    for (int axis = 0; axis < 3; ++axis) {
        scene->obstacle[2 * axis] = 10.0f;
        scene->obstacle[2 * axis + 1] = 11.0f;
    }
    scene->detour[0] = scene->target[0];
    scene->detour[1] = scene->target[1];
    scene->detour_active = 0;
}

static int configure_coverage(
    const float *position,
    const float *room,
    FlightRLDoorScene *scene
) {
    if (scene->obstacle[0] > 5.0f) {
        scene->coverage[0] = 0.5f * (room[0] + room[1]);
        scene->coverage[1] = 0.5f * (room[2] + room[3]);
        return 1;
    }
    const float margin = FLIGHTRL_DOOR_ROUTE_CLEARANCE_M;
    float obstacle_x = 0.5f * (scene->obstacle[0] + scene->obstacle[1]);
    float obstacle_y = 0.5f * (scene->obstacle[2] + scene->obstacle[3]);
    float dx = scene->target[0] - position[0];
    float dy = scene->target[1] - position[1];
    float inv = 1.0f / fmaxf(hypotf(dx, dy), 1.0e-3f);
    float lateral_x = -dy * inv;
    float lateral_y = dx * inv;
    float half_diagonal = hypotf(
        0.5f * (scene->obstacle[1] - scene->obstacle[0]),
        0.5f * (scene->obstacle[3] - scene->obstacle[2])
    );
    float best_length = INFINITY;
    int found = 0;
    for (int ring = 0; ring < 5; ++ring) {
        float offset = half_diagonal + 0.35f + 0.20f * (float)ring;
        for (int side = 0; side < 2; ++side) {
            float sign = side == 0 ? -1.0f : 1.0f;
            float candidate[3] = {
                obstacle_x + sign * lateral_x * offset,
                obstacle_y + sign * lateral_y * offset,
                position[2],
            };
            float approach_dx = candidate[0] - position[0];
            float approach_dy = candidate[1] - position[1];
            float approach_inverse = 1.0f / fmaxf(
                hypotf(approach_dx, approach_dy),
                1.0e-3f
            );
            float turn_entry[3] = {
                candidate[0]
                    - FLIGHTRL_DOOR_DETOUR_RELEASE_RADIUS_M
                        * approach_dx * approach_inverse,
                candidate[1]
                    - FLIGHTRL_DOOR_DETOUR_RELEASE_RADIUS_M
                        * approach_dy * approach_inverse,
                position[2],
            };
            float center[3];
            door_center(room, scene, center);
            if (
                candidate[0] < room[0] + 0.30f
                || candidate[0] > room[1] - 0.30f
                || candidate[1] < room[2] + 0.30f
                || candidate[1] > room[3] - 0.30f
                || point_in_obstacle(candidate, scene, margin)
                || segment_hits_obstacle(position, candidate, scene, margin)
                || segment_hits_obstacle(candidate, scene->target, scene, margin)
                || segment_hits_obstacle(turn_entry, scene->target, scene, margin)
                || segment_hits_obstacle(candidate, center, scene, 0.0f)
            ) continue;
            float length = hypotf(
                candidate[0] - position[0],
                candidate[1] - position[1]
            ) + hypotf(
                scene->target[0] - candidate[0],
                scene->target[1] - candidate[1]
            );
            if (length < best_length) {
                best_length = length;
                scene->coverage[0] = candidate[0];
                scene->coverage[1] = candidate[1];
                scene->teacher_side = sign;
                found = 1;
            }
        }
    }
    return found;
}

static void sample_obstacle(
    const float *position,
    const float *room,
    FlightRLDoorScene *scene,
    uint32_t *rng,
    float layout_diversity
) {
    const float margin = FLIGHTRL_DOOR_COLLISION_MARGIN_M;
    disable_obstacle(scene);
    for (int attempt = 0; attempt < 128; ++attempt) {
        float mix = rnd(rng, 0.30f, 0.72f);
        float center_x = position[0] + mix * (scene->target[0] - position[0]);
        float center_y = position[1] + mix * (scene->target[1] - position[1]);
        float half_x = rnd(rng, 0.16f, 0.30f);
        float half_y = rnd(rng, 0.16f, 0.30f);
        float obstacle_height = rnd(rng, 1.0f, 1.65f);
        int family = layout_diversity > 0.5f
            ? (int)rnd(rng, 0.0f, 3.999999f)
            : 0;
        if (family == 1) {
            half_x = rnd(rng, 0.22f, 0.38f);
            half_y = rnd(rng, 0.42f, 0.72f);
            obstacle_height = rnd(rng, 1.35f, 2.05f);
        } else if (family == 2) {
            int along_x = rnd(rng, 0.0f, 1.0f) < 0.5f;
            half_x = along_x
                ? rnd(rng, 0.65f, 1.05f)
                : rnd(rng, 0.08f, 0.14f);
            half_y = along_x
                ? rnd(rng, 0.08f, 0.14f)
                : rnd(rng, 0.65f, 1.05f);
            obstacle_height = room[5] - room[4] - 0.08f;
        } else if (family == 3) {
            half_x = rnd(rng, 0.12f, 0.22f);
            half_y = rnd(rng, 0.12f, 0.22f);
            obstacle_height = room[5] - room[4] - 0.04f;
        }
        center_x = clampf(
            center_x,
            room[0] + half_x + 0.08f,
            room[1] - half_x - 0.08f
        );
        center_y = clampf(
            center_y,
            room[2] + half_y + 0.08f,
            room[3] - half_y - 0.08f
        );
        scene->obstacle[0] = center_x - half_x;
        scene->obstacle[1] = center_x + half_x;
        scene->obstacle[2] = center_y - half_y;
        scene->obstacle[3] = center_y + half_y;
        scene->obstacle[4] = room[4];
        scene->obstacle[5] = fminf(
            room[5] - 0.02f,
            room[4] + obstacle_height
        );
        if (
            point_in_obstacle(position, scene, margin)
            || point_in_obstacle(scene->target, scene, margin)
            || !configure_coverage(position, room, scene)
        ) continue;
        scene->detour[0] = scene->coverage[0];
        scene->detour[1] = scene->coverage[1];
        scene->detour_active = 1;
        return;
    }
    disable_obstacle(scene);
    configure_coverage(position, room, scene);
}

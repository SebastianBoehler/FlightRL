#include <math.h>

#include "native_door_scene.h"

#define DOOR_PI 3.14159265358979323846f

static float door_yaw(const float *q) {
    return atan2f(
        2.0f * (q[0] * q[3] + q[1] * q[2]),
        1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3])
    );
}

static float wrap_door_angle(float value) {
    while (value > DOOR_PI) value -= 2.0f * DOOR_PI;
    while (value < -DOOR_PI) value += 2.0f * DOOR_PI;
    return value;
}

static void door_center(
    const float *room,
    const FlightRLDoorScene *scene,
    float *center
) {
    int face = (int)scene->door[0];
    center[0] = face == 0 ? room[0] : (face == 1 ? room[1] : scene->door[1]);
    center[1] = face == 2 ? room[2] : (face == 3 ? room[3] : scene->door[1]);
    center[2] = 0.5f * (scene->door[3] + scene->door[4]);
}

static int point_in_obstacle(
    const float *position,
    const FlightRLDoorScene *scene,
    float margin
) {
    if (scene->obstacle[0] > 5.0f) return 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (
            position[axis] < scene->obstacle[2 * axis] - margin
            || position[axis] > scene->obstacle[2 * axis + 1] + margin
        ) return 0;
    }
    return 1;
}

static int segment_hits_obstacle(
    const float *origin,
    const float *target,
    const FlightRLDoorScene *scene,
    float margin
) {
    if (scene->obstacle[0] > 5.0f) return 0;
    float near = 0.0f;
    float far = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
        float direction = target[axis] - origin[axis];
        float low = scene->obstacle[2 * axis] - margin;
        float high = scene->obstacle[2 * axis + 1] + margin;
        if (fabsf(direction) < 1.0e-6f) {
            if (origin[axis] < low || origin[axis] > high) return 0;
            continue;
        }
        float first = (low - origin[axis]) / direction;
        float second = (high - origin[axis]) / direction;
        if (first > second) {
            float swap = first;
            first = second;
            second = swap;
        }
        near = fmaxf(near, first);
        far = fminf(far, second);
        if (near > far) return 0;
    }
    return near < 1.0f && far > 0.0f;
}

#include "native_door_scene_coverage.inc"

void flightrl_door_scene_sample(
    float *position,
    float *quaternion,
    const float *room,
    FlightRLDoorScene *scene,
    uint32_t *rng,
    float obstacle_probability,
    float layout_diversity
) {
    int face = (int)(4.0f * rnd(rng, 0.0f, 0.999999f));
    float tangent_min = face < 2 ? room[2] : room[0];
    float tangent_max = face < 2 ? room[3] : room[1];
    scene->door[0] = (float)face;
    scene->door[2] = rnd(rng, 0.65f, 0.95f);
    scene->door[1] = rnd(
        rng,
        tangent_min + 0.55f * scene->door[2],
        tangent_max - 0.55f * scene->door[2]
    );
    scene->door[3] = room[4];
    scene->door[4] = rnd(rng, 1.75f, fminf(2.25f, room[5] - 0.05f));
    float inward_x = face == 0 ? 1.0f : (face == 1 ? -1.0f : 0.0f);
    float inward_y = face == 2 ? 1.0f : (face == 3 ? -1.0f : 0.0f);
    float center[3];
    door_center(room, scene, center);
    scene->target[0] = center[0] + 0.52f * inward_x;
    scene->target[1] = center[1] + 0.52f * inward_y;
    scene->target[2] = rnd(rng, 0.65f, 1.05f);
    scene->target_yaw = atan2f(-inward_y, -inward_x);

    for (int attempt = 0; attempt < 64; ++attempt) {
        position[0] = rnd(rng, room[0] + 0.35f, room[1] - 0.35f);
        position[1] = rnd(rng, room[2] + 0.35f, room[3] - 0.35f);
        position[2] = scene->target[2];
        if (flightrl_door_scene_distance(position, scene) > 1.25f) break;
    }
    float bearing = atan2f(
        scene->target[1] - position[1],
        scene->target[0] - position[0]
    );
    float yaw;
    if (rnd(rng, 0.0f, 1.0f) < 0.5f) {
        yaw = bearing + rnd(rng, -0.35f, 0.35f);
    } else {
        float sign = rnd(rng, 0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f;
        yaw = bearing + sign * rnd(rng, 1.15f, DOOR_PI);
    }
    quaternion[0] = cosf(0.5f * yaw);
    quaternion[1] = 0.0f;
    quaternion[2] = 0.0f;
    quaternion[3] = sinf(0.5f * yaw);

    for (int i = 0; i < 3; ++i) {
        scene->obstacle[2 * i] = 10.0f;
        scene->obstacle[2 * i + 1] = 11.0f;
    }
    scene->teacher_side = rnd(rng, 0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f;
    scene->detour[0] = scene->target[0];
    scene->detour[1] = scene->target[1];
    scene->detour_active = 0;
    if (rnd(rng, 0.0f, 1.0f) < obstacle_probability) {
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
        float dx = scene->target[0] - position[0];
        float dy = scene->target[1] - position[1];
        float inv = 1.0f / fmaxf(sqrtf(dx * dx + dy * dy), 1.0e-3f);
        float detour = fmaxf(half_x, half_y) + 0.75f;
        scene->detour[0] = center_x - scene->teacher_side * dy * inv * detour;
        scene->detour[1] = center_y + scene->teacher_side * dx * inv * detour;
        if (
            scene->detour[0] < room[0] + 0.25f
            || scene->detour[0] > room[1] - 0.25f
            || scene->detour[1] < room[2] + 0.25f
            || scene->detour[1] > room[3] - 0.25f
        ) {
            scene->teacher_side *= -1.0f;
            scene->detour[0] = center_x - scene->teacher_side * dy * inv * detour;
            scene->detour[1] = center_y + scene->teacher_side * dx * inv * detour;
        }
        scene->detour[0] = clampf(scene->detour[0], room[0] + 0.25f, room[1] - 0.25f);
        scene->detour[1] = clampf(scene->detour[1], room[2] + 0.25f, room[3] - 0.25f);
        scene->detour_active = 1;
    }
    configure_coverage(position, room, scene);
    if (scene->obstacle[0] <= 5.0f) {
        scene->detour[0] = scene->coverage[0];
        scene->detour[1] = scene->coverage[1];
    }
    scene->initial_outside_fov = !flightrl_door_scene_visible(
        position,
        quaternion,
        room,
        scene
    );
    scene->target_observed = !scene->initial_outside_fov;
    scene->search_yaw = door_yaw(quaternion);
    scene->search_yaw_progress = 0.0f;
    scene->search_phase = 0;
}

float flightrl_door_scene_distance(
    const float *position,
    const FlightRLDoorScene *scene
) {
    float dx = scene->target[0] - position[0];
    float dy = scene->target[1] - position[1];
    return sqrtf(dx * dx + dy * dy);
}

int flightrl_door_scene_collides(
    const float *position,
    const float *room,
    const FlightRLDoorScene *scene
) {
    float margin = 0.07f;
    int wall = (
        position[0] < room[0] + margin || position[0] > room[1] - margin
        || position[1] < room[2] + margin || position[1] > room[3] - margin
        || position[2] < room[4] + margin || position[2] > room[5] - margin
    );
    return wall || point_in_obstacle(position, scene, margin);
}

int flightrl_door_scene_visible(
    const float *position,
    const float *quaternion,
    const float *room,
    const FlightRLDoorScene *scene
) {
    float center[3];
    door_center(room, scene, center);
    float bearing = wrap_door_angle(
        atan2f(center[1] - position[1], center[0] - position[0])
        - door_yaw(quaternion)
    );
    if (fabsf(bearing) > 0.7156f) return 0;
    return !segment_hits_obstacle(position, center, scene, 0.0f);
}

float flightrl_door_scene_center_x(
    const float *position,
    const float *quaternion,
    const float *room,
    const FlightRLDoorScene *scene
) {
    float center[3];
    door_center(room, scene, center);
    float bearing = wrap_door_angle(
        atan2f(center[1] - position[1], center[0] - position[0])
        - door_yaw(quaternion)
    );
    float tan_x = tanf(0.5f * 1.099557429f) * 4.0f / 3.0f;
    return clampf(0.5f - 0.5f * tanf(bearing) / tan_x, 0.0f, 1.0f);
}

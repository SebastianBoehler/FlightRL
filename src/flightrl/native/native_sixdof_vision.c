#include <math.h>
#include <stdint.h>
#include <string.h>
#include "native_sixdof_vision.h"
#define VISION_FOV_Y_RAD 1.099557429f
#define VISION_MAX_DISTANCE 8.0f
static float vision_clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static void vision_quat_matrix(const float *q, float r[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    r[0] = 1.0f - 2.0f * (y * y + z * z);
    r[1] = 2.0f * (x * y - z * w);
    r[2] = 2.0f * (x * z + y * w);
    r[3] = 2.0f * (x * y + z * w);
    r[4] = 1.0f - 2.0f * (x * x + z * z);
    r[5] = 2.0f * (y * z - x * w);
    r[6] = 2.0f * (x * z - y * w);
    r[7] = 2.0f * (y * z + x * w);
    r[8] = 1.0f - 2.0f * (x * x + y * y);
}

static uint32_t vision_hash(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

static float intersect_room(
    const float *origin,
    const float *direction,
    const float *room,
    int *face
) {
    const float low[3] = {room[0], room[2], room[4]};
    const float high[3] = {room[1], room[3], room[5]};
    float best = VISION_MAX_DISTANCE;
    *face = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(direction[axis]) < 1.0e-6f) {
            continue;
        }
        int side = direction[axis] > 0.0f;
        float plane = side ? high[axis] : low[axis];
        float distance = (plane - origin[axis]) / direction[axis];
        if (distance > 1.0e-5f && distance < best) {
            best = distance;
            *face = axis * 2 + side;
        }
    }
    return best;
}

static float intersect_box(const float *origin, const float *direction, const float *box) {
    float near = 0.0f;
    float far = VISION_MAX_DISTANCE;
    for (int axis = 0; axis < 3; ++axis) {
        float low = box[2 * axis];
        float high = box[2 * axis + 1];
        if (fabsf(direction[axis]) < 1.0e-6f) {
            if (origin[axis] < low || origin[axis] > high) {
                return VISION_MAX_DISTANCE;
            }
            continue;
        }
        float first = (low - origin[axis]) / direction[axis];
        float second = (high - origin[axis]) / direction[axis];
        if (first > second) {
            float swap = first;
            first = second;
            second = swap;
        }
        near = fmaxf(near, first);
        far = fminf(far, second);
        if (near > far) {
            return VISION_MAX_DISTANCE;
        }
    }
    return near > 1.0e-5f ? near : VISION_MAX_DISTANCE;
}

static float surface_intensity(const float *hit, int face, float distance, int scene_seed, int obstacle_hit) {
    if (obstacle_hit) {
        int pattern = ((int)floorf(hit[0] * 6.0f) + (int)floorf(hit[1] * 6.0f) + (int)floorf(hit[2] * 6.0f)) & 1;
        return pattern ? 38.0f : 222.0f;
    }
    float base;
    if (face < 2) {
        base = 208.0f;
    } else if (face < 4) {
        base = 174.0f;
    } else {
        base = face == 4 ? 150.0f : 192.0f;
    }
    int checker = ((int)floorf((hit[0] + 2.0f) * 2.0f) + (int)floorf((hit[1] + 2.0f) * 2.0f)) & 1;
    if (face == 4) {
        base += checker ? 22.0f : -16.0f;
    } else if (face < 4) {
        base += 12.0f * sinf(4.0f * hit[2] + 0.17f * (float)(scene_seed & 7));
    }
    if (face == 1 && hit[1] > 0.30f && hit[1] < 1.10f && hit[2] > 0.65f && hit[2] < 1.60f) {
        base = 42.0f;
    }
    if (face == 3 && hit[0] > -1.15f && hit[0] < -0.25f && hit[2] > 0.50f && hit[2] < 1.10f) {
        base = 232.0f;
    }

    float attenuation = vision_clamp(1.02f - 0.055f * distance, 0.70f, 1.0f);
    return vision_clamp(base * attenuation, 0.0f, 255.0f);
}

static void render_one(
    const float *position,
    const float *quaternion,
    const float *room,
    const float *obstacle,
    float target_mean,
    int scene_seed,
    uint8_t *frame
) {
    float rotation[9];
    vision_quat_matrix(quaternion, rotation);
    float origin[3] = {
        position[0] + 0.035f * rotation[0] + 0.012f * rotation[2],
        position[1] + 0.035f * rotation[3] + 0.012f * rotation[5],
        position[2] + 0.035f * rotation[6] + 0.012f * rotation[8],
    };
    origin[0] = vision_clamp(origin[0], room[0] + 0.005f, room[1] - 0.005f);
    origin[1] = vision_clamp(origin[1], room[2] + 0.005f, room[3] - 0.005f);
    origin[2] = vision_clamp(origin[2], room[4] + 0.005f, room[5] - 0.005f);

    const float tan_y = tanf(0.5f * VISION_FOV_Y_RAD);
    const float tan_x = tan_y * ((float)SIXDOF_VISION_WIDTH / (float)SIXDOF_VISION_HEIGHT);
    float sum = 0.0f;
    for (int row = 0; row < SIXDOF_VISION_HEIGHT; ++row) {
        float screen_y = (2.0f * ((float)row + 0.5f) / SIXDOF_VISION_HEIGHT - 1.0f) * tan_y;
        for (int col = 0; col < SIXDOF_VISION_WIDTH; ++col) {
            float screen_x = (2.0f * ((float)col + 0.5f) / SIXDOF_VISION_WIDTH - 1.0f) * tan_x;
            float body[3] = {1.0f, -screen_x, -screen_y};
            float inv_norm = 1.0f / sqrtf(1.0f + screen_x * screen_x + screen_y * screen_y);
            float direction[3] = {
                inv_norm * (rotation[0] * body[0] + rotation[1] * body[1] + rotation[2] * body[2]),
                inv_norm * (rotation[3] * body[0] + rotation[4] * body[1] + rotation[5] * body[2]),
                inv_norm * (rotation[6] * body[0] + rotation[7] * body[1] + rotation[8] * body[2]),
            };
            int face;
            float distance = intersect_room(origin, direction, room, &face);
            float obstacle_distance = obstacle ? intersect_box(origin, direction, obstacle) : VISION_MAX_DISTANCE;
            int obstacle_hit = obstacle_distance < distance;
            if (obstacle_hit) {
                distance = obstacle_distance;
            }
            float hit[3] = {
                origin[0] + distance * direction[0],
                origin[1] + distance * direction[1],
                origin[2] + distance * direction[2],
            };
            int pixel = row * SIXDOF_VISION_WIDTH + col;
            frame[pixel] = (uint8_t)surface_intensity(hit, face, distance, scene_seed, obstacle_hit);
            sum += frame[pixel];
        }
    }

    float scale = vision_clamp(target_mean, 20.0f, 140.0f) * SIXDOF_VISION_PIXELS / fmaxf(sum, 1.0f);
    for (int pixel = 0; pixel < SIXDOF_VISION_PIXELS; ++pixel) {
        uint32_t noise_hash = vision_hash((uint32_t)scene_seed ^ (uint32_t)pixel);
        float noise = ((float)(noise_hash & 255u) / 255.0f - 0.5f) * 8.0f;
        float adjusted = vision_clamp(frame[pixel] * scale + noise, 0.0f, 255.0f);
        frame[pixel] = (uint8_t)(17.0f * floorf(adjusted / 17.0f + 0.5f));
    }
}

void flightrl_sixdof_render_gray4_batch(
    const float *position,
    const float *quaternion,
    const float *room,
    const float *target_mean,
    const int *scene_seed,
    uint8_t *frames,
    int num_envs
) {
    for (int env = 0; env < num_envs; ++env) {
        render_one(
            position + env * 3,
            quaternion + env * 4,
            room,
            NULL,
            target_mean[env],
            scene_seed[env],
            frames + env * SIXDOF_VISION_PIXELS
        );
    }
}

static float visual_yaw(const float *q) {
    return atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]), 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]));
}

static void assemble_visual_observation(
    const float *position,
    const float *quaternion,
    const float *target_position,
    float target_yaw,
    const uint8_t *current,
    uint8_t *previous,
    uint8_t reset_temporal,
    float *observation
) {
    for (int pixel = 0; pixel < SIXDOF_VISION_PIXELS; ++pixel) {
        float gray = current[pixel] / 255.0f;
        float delta = reset_temporal ? 0.0f : (current[pixel] - previous[pixel]) / 255.0f;
        observation[pixel] = 2.0f * gray - 1.0f;
        observation[SIXDOF_VISION_PIXELS + pixel] = delta;
        observation[2 * SIXDOF_VISION_PIXELS + pixel] = fabsf(delta) >= 0.08f ? 1.0f : 0.0f;
    }
    memcpy(previous, current, SIXDOF_VISION_PIXELS);

    float rotation[9];
    vision_quat_matrix(quaternion, rotation);
    float world[3] = {
        target_position[0] - position[0],
        target_position[1] - position[1],
        target_position[2] - position[2],
    };
    float body[3] = {
        rotation[0] * world[0] + rotation[3] * world[1] + rotation[6] * world[2],
        rotation[1] * world[0] + rotation[4] * world[1] + rotation[7] * world[2],
        rotation[2] * world[0] + rotation[5] * world[1] + rotation[8] * world[2],
    };
    float distance = sqrtf(body[0] * body[0] + body[1] * body[1] + body[2] * body[2]);
    float inv_distance = distance > 1.0e-6f ? 1.0f / distance : 0.0f;
    int offset = SIXDOF_VISION_CHANNELS * SIXDOF_VISION_PIXELS;
    observation[offset] = body[0] * inv_distance;
    observation[offset + 1] = body[1] * inv_distance;
    observation[offset + 2] = body[2] * inv_distance;
    observation[offset + 3] = vision_clamp(distance / 4.0f, 0.0f, 1.0f);
    float yaw_error = target_yaw - visual_yaw(quaternion);
    observation[offset + 4] = sinf(yaw_error);
    observation[offset + 5] = cosf(yaw_error);
}

void flightrl_sixdof_visual_observation_batch(
    const float *position,
    const float *quaternion,
    const float *target_position,
    const float *target_yaw,
    const float *room,
    const float *target_mean,
    const int *scene_seed,
    uint8_t *previous_frame,
    const uint8_t *reset_temporal,
    float *observations,
    int num_envs
) {
    uint8_t current[SIXDOF_VISION_PIXELS];
    for (int env = 0; env < num_envs; ++env) {
        render_one(position + env * 3, quaternion + env * 4, room, NULL, target_mean[env], scene_seed[env], current);
        assemble_visual_observation(
            position + env * 3,
            quaternion + env * 4,
            target_position + env * 3,
            target_yaw[env],
            current,
            previous_frame + env * SIXDOF_VISION_PIXELS,
            reset_temporal[env],
            observations + env * SIXDOF_VISION_OBS_DIM
        );
    }
}

void flightrl_sixdof_visual_observation_scene(
    const float *position,
    const float *quaternion,
    const float *target_position,
    float target_yaw,
    const float *room,
    const float *obstacle,
    float target_mean,
    int scene_seed,
    uint8_t *previous_frame,
    uint8_t reset_temporal,
    float *observation
) {
    uint8_t current[SIXDOF_VISION_PIXELS];
    render_one(position, quaternion, room, obstacle, target_mean, scene_seed, current);
    assemble_visual_observation(
        position,
        quaternion,
        target_position,
        target_yaw,
        current,
        previous_frame,
        reset_temporal,
        observation
    );
}

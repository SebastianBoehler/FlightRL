static uint32_t vision_hash(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

static float vision_unit(uint32_t value) {
    return (float)(value & 0xffffu) / 65535.0f;
}

static int door_surface(
    const float *hit,
    int face,
    const float *door,
    float *intensity
) {
    if (!door || face != (int)door[0]) return 0;
    float tangent = face < 2 ? hit[1] : hit[0];
    float x = (tangent - door[1]) / door[2] + 0.5f;
    float y = (hit[2] - door[3]) / fmaxf(door[4] - door[3], 1.0e-3f);
    if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) return 0;
    uint32_t style = vision_hash((uint32_t)door[5]);
    float panel = 45.0f + (float)(style & 127u);
    float trim = 20.0f + (float)((style >> 8) & 47u);
    int edge = x < 0.055f || x > 0.945f || y < 0.035f || y > 0.965f;
    int inset = (
        x > 0.16f && x < 0.84f
        && ((y > 0.14f && y < 0.43f) || (y > 0.57f && y < 0.86f))
    );
    float handle_dx = x - 0.79f;
    float handle_dy = y - 0.50f;
    int handle = handle_dx * handle_dx + handle_dy * handle_dy < 0.0020f;
    *intensity = handle ? 230.0f : (edge ? trim : panel + (inset ? 24.0f : 0.0f));
    return 1;
}

static void distractor_dimensions(
    int style,
    float wall_width,
    float ceiling,
    float *width,
    float *bottom,
    float *top
) {
    if (style == 0) {
        *width = fminf(1.45f, 0.38f * wall_width);
        *bottom = 0.42f;
        *top = fminf(2.15f, ceiling - 0.12f);
    } else if (style == 1) {
        *width = fminf(1.70f, 0.44f * wall_width);
        *bottom = 0.12f;
        *top = fminf(1.28f, ceiling - 0.18f);
    } else if (style == 2) {
        *width = fminf(1.25f, 0.34f * wall_width);
        *bottom = 0.52f;
        *top = fminf(1.62f, ceiling - 0.18f);
    } else if (style == 3) {
        *width = fminf(0.88f, 0.27f * wall_width);
        *bottom = 0.0f;
        *top = fminf(2.30f, ceiling);
    } else {
        *width = fminf(1.12f, 0.32f * wall_width);
        *bottom = 0.30f;
        *top = fminf(1.88f, ceiling - 0.12f);
    }
}

static float distractor_intensity(
    int style,
    float x,
    float y,
    uint32_t material
) {
    int edge = x < 0.045f || x > 0.955f || y < 0.035f || y > 0.965f;
    if (style == 0) {
        int mullion = (
            fabsf(x - 0.5f) < 0.025f
            || fabsf(y - 0.58f) < 0.020f
        );
        float pane = ((int)floorf(4.0f * x) & 1) ? 62.0f : 188.0f;
        return edge || mullion ? 28.0f : pane;
    }
    if (style == 1) {
        int seam = fabsf(4.0f * x - floorf(4.0f * x + 0.5f)) < 0.035f;
        int counter = fabsf(y - 0.88f) < 0.045f;
        int oven = x > 0.25f && x < 0.75f && y > 0.12f && y < 0.68f;
        int handle = y > 0.72f && y < 0.76f && fmodf(4.0f * x, 1.0f) > 0.25f;
        if (oven) return edge ? 132.0f : 22.0f;
        if (handle) return 226.0f;
        return edge || seam || counter ? 112.0f : 174.0f;
    }
    if (style == 2) {
        int screen = x > 0.12f && x < 0.88f && y > 0.30f && y < 0.92f;
        int stand = fabsf(x - 0.5f) < 0.055f && y > 0.12f && y < 0.30f;
        int desk = y < 0.14f;
        if (screen) return edge ? 118.0f : 18.0f + 22.0f * vision_unit(material);
        return stand ? 54.0f : (desk ? 176.0f : 202.0f);
    }
    if (style == 3) {
        float depth_gradient = 18.0f + 74.0f * y;
        if (material & 1u) depth_gradient = 230.0f - depth_gradient;
        return edge ? 126.0f : depth_gradient;
    }
    int shelf = fabsf(5.0f * y - floorf(5.0f * y + 0.5f)) < 0.045f;
    int upright = fabsf(x - 0.5f) < 0.025f;
    int item = (
        fmodf(7.0f * x + 3.0f * y, 1.0f) > 0.62f
        && !shelf
    );
    return edge || shelf || upright ? 72.0f : (item ? 28.0f : 194.0f);
}

static int wall_distractor(
    const float *hit,
    int face,
    const float *room,
    int scene_seed,
    float *intensity
) {
    if (face >= 4) return 0;
    float low = face < 2 ? room[2] : room[0];
    float high = face < 2 ? room[3] : room[1];
    float tangent = face < 2 ? hit[1] : hit[0];
    float wall_width = high - low;
    float ceiling = room[5] - room[4];
    uint32_t grammar = vision_hash(
        (uint32_t)scene_seed ^ (uint32_t)(face + 11) * 0x85ebca6bu
    );
    int count = 3 + (int)(grammar & 1u);
    for (int slot = count - 1; slot >= 0; --slot) {
        uint32_t material = vision_hash(
            grammar ^ (uint32_t)(slot + 1) * 0x9e3779b9u
        );
        int style = (int)((grammar + 3u * (uint32_t)slot) % 5u);
        float width, bottom, top;
        distractor_dimensions(style, wall_width, ceiling, &width, &bottom, &top);
        float fraction = ((float)slot + 0.5f) / (float)count;
        float jitter = (vision_unit(material) - 0.5f) * 0.18f / (float)count;
        float center = low + wall_width * vision_clamp(
            fraction + jitter,
            0.08f,
            0.92f
        );
        float x = (tangent - center) / width + 0.5f;
        float y = (hit[2] - room[4] - bottom) / fmaxf(top - bottom, 1.0e-3f);
        if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) continue;
        *intensity = distractor_intensity(style, x, y, material);
        return 1;
    }
    return 0;
}

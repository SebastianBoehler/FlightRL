#ifndef FLIGHTRL_INSPECTION_SCENE_H
#define FLIGHTRL_INSPECTION_SCENE_H
#include <stdint.h>
/* Panel rows: center xyz, unit u xyz, unit v xyz, half width/height, RGB.
 * Normal = u cross v, visible from normal side. Metres, FLU, wxyz.
 * Shared boxes: xmin,xmax,ymin,ymax,zmin,zmax. No evaluator IDs here. */
void flightrl_inspection_render(const float *positions, const float *quaternions,
    const float *room, const float *boxes, int box_count, const float *panels,
    int panel_count, int count, uint8_t *frames, int32_t *counts);
void flightrl_inspection_render_depth(const float *positions, const float *quaternions,
    const float *room, const float *boxes, int box_count, const float *panels,
    int panel_count, int count, uint8_t *frames, int32_t *counts, float *depth);
void flightrl_inspection_render_sized(const float *positions,const float *quaternions,
    const float *room,const float *boxes,int box_count,const float *panels,int panel_count,
    int count,uint8_t *frames,int32_t *counts,float *depth,int width,int height,int materials,
    const float *appearance,const float *lights,int light_count,const float *windows,int window_count);
/* Conservative swept axis-aligned body envelope, contact terminates episode.
 * This is collision detection, not rigid-body contact response. */
void flightrl_inspection_collision(const float *start, const float *end,
    const float *room, const float *boxes, int box_count, int count,
    float radius, uint8_t *collisions);
#endif

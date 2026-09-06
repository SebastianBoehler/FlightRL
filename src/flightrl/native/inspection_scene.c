#include <math.h>
#include <string.h>
#include "inspection_scene.h"

static float dot(const float *a, const float *b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

static int box_interval(const float *o, const float *d, const float *box,
                        float radius, float *near, float *far) {
    for (int k=0; k<3; ++k) {
        float lo=box[2*k]-radius, hi=box[2*k+1]+radius;
        if (fabsf(d[k])<1e-8f) {
            if (o[k]<lo || o[k]>hi) return 0;
        } else {
            float a=(lo-o[k])/d[k], b=(hi-o[k])/d[k];
            *near=fmaxf(*near,fminf(a,b));
            *far=fminf(*far,fmaxf(a,b));
            if (*near>*far) return 0;
        }
    }
    return 1;
}

static void rotation(const float *q, float *r) {
    float w=q[0],x=q[1],y=q[2],z=q[3];
    r[0]=1-2*(y*y+z*z); r[1]=2*(x*y-z*w); r[2]=2*(x*z+y*w);
    r[3]=2*(x*y+z*w); r[4]=1-2*(x*x+z*z); r[5]=2*(y*z-x*w);
    r[6]=2*(x*z-y*w); r[7]=2*(y*z+x*w); r[8]=1-2*(x*x+y*y);
}

static float panel_hit(const float *o, const float *d, const float *p) {
    const float *u=p+3,*v=p+6;
    float n[3]={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    float denom=dot(d,n), delta[3]={p[0]-o[0],p[1]-o[1],p[2]-o[2]};
    if (denom>=-1e-6f) return INFINITY;
    float t=dot(delta,n)/denom;
    if (t<=0 || t>8) return INFINITY;
    float hit[3]={o[0]+t*d[0]-p[0],o[1]+t*d[1]-p[1],o[2]+t*d[2]-p[2]};
    return fabsf(dot(hit,u))<=p[9] && fabsf(dot(hit,v))<=p[10] ? t : INFINITY;
}

#include "inspection_forest.inc"
#include "inspection_materials.inc"

void flightrl_inspection_render_sized(const float *positions, const float *quaternions,
    const float *room, const float *boxes, int box_count, const float *panels,
    int panel_count, int count, uint8_t *frames, int32_t *counts, float *depth, int width, int height, int materials, const float *appearance,const float *lights,int light_count,const float *windows,int window_count) {
    memset(counts,0,(size_t)count*panel_count*2*sizeof(int32_t));
    float fy=tanf(1.099557429f/2);
    for (int e=0;e<count;++e) {
        float r[9],o[3]; rotation(quaternions+4*e,r);
        for (int k=0;k<3;++k) o[k]=positions[3*e+k]+.035f*r[3*k]+.012f*r[3*k+2];
        for (int row=0;row<height;++row) for (int col=0;col<width;++col) {
            float b[3]={1,-(2*(col+.5f)/width-1)*fy*width/height,-(2*(row+.5f)/height-1)*fy};
            float d[3];
            for (int k=0;k<3;++k) d[k]=dot(r+3*k,b);
            float inv=1/sqrtf(dot(d,d)); for (int k=0;k<3;++k) d[k]*=inv;
            float near=0,far=8,best=8;
            if (box_interval(o,d,room,0,&near,&far)) best=far;
            int obstacle=0, winner=-1, box_index=-1;
            for (int j=0;j<box_count;++j) {
                near=0; far=8;
                if ((materials && appearance[20]==2 ? forest_hit(o,d,boxes+6*j,&near) : box_interval(o,d,boxes+6*j,0,&near,&far)) && near<best) {
                    best=near; obstacle=1; box_index=j;
                }
            }
            for (int j=0;j<panel_count;++j) {
                float t=panel_hit(o,d,panels+14*j);
                if (isfinite(t)) counts[(e*panel_count+j)*2+1]++;
                if (t<best) {best=t; winner=j;}
            }
            if (depth) depth[(e*height+row)*width+col]=best;
            uint8_t *pixel=frames+((e*height+row)*width+col)*3;
            if (winner>=0) {
                counts[(e*panel_count+winner)*2]++;
                for (int k=0;k<3;++k) pixel[k]=(uint8_t)panels[14*winner+11+k];
            } else {
                /* Neutral authored surfaces; marker palette is an explicit diagnostic assumption. */
                uint8_t gray=obstacle ? 65 : (uint8_t)(100+20*((row/8+col/8)%2));
                pixel[0]=pixel[1]=pixel[2]=gray;
            }
            if(materials) industrial_surface(o,d,best,room,boxes,box_count,box_index,winner,panels,pixel,appearance,lights,light_count,windows,window_count);
        }
    }
}

void flightrl_inspection_collision(const float *start, const float *end,
    const float *room, const float *boxes, int box_count, int count,
    float radius, uint8_t *collisions) {
    for (int e=0;e<count;++e) {
        const float *a=start+3*e,*b=end+3*e;
        float d[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
        collisions[e]=0;
        for (int k=0;k<3;++k)
            if (a[k]<=room[2*k]+radius || a[k]>=room[2*k+1]-radius ||
                b[k]<=room[2*k]+radius || b[k]>=room[2*k+1]-radius) collisions[e]=1;
        for (int j=0;j<box_count;++j) {
            float near=0,far=1;
            if (box_interval(a,d,boxes+6*j,radius,&near,&far)) collisions[e]=1;
        }
    }
}

void flightrl_inspection_render(const float *positions, const float *quaternions,
    const float *room, const float *boxes, int box_count, const float *panels,
    int panel_count, int count, uint8_t *frames, int32_t *counts) {
    flightrl_inspection_render_depth(positions,quaternions,room,boxes,box_count,
                                    panels,panel_count,count,frames,counts,NULL);
}

void flightrl_inspection_render_depth(const float *p,const float *q,const float *room,
    const float *boxes,int nb,const float *panels,int np,int n,uint8_t *rgb,int32_t *counts,float *depth) {
    flightrl_inspection_render_sized(p,q,room,boxes,nb,panels,np,n,rgb,counts,depth,64,48,0,NULL,NULL,0,NULL,0);
}

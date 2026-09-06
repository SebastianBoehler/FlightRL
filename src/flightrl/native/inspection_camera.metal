#include <metal_stdlib>
using namespace metal;

bool interval(float3 o,float3 d,device const float* box,thread float& lo,thread float& hi) {
    for(int k=0;k<3;k++) {
        if(abs(d[k])<1e-8f) {if(o[k]<box[2*k] || o[k]>box[2*k+1]) return false;}
        else {
            float a=(box[2*k]-o[k])/d[k],b=(box[2*k+1]-o[k])/d[k];
            lo=max(lo,min(a,b)); hi=min(hi,max(a,b)); if(lo>hi) return false;
        }
    }
    return true;
}
float3 load3(device const float* p) {return float3(p[0],p[1],p[2]);}

kernel void camera(device const float* positions,device const float* qs,
    device const float* room,device const float* boxes,device const float* panels,
    device uchar* rgb,device float* depth,constant int& nb,constant int& np,
    uint idx [[thread_position_in_grid]]) {
    uint e=idx/3072,pixel=idx%3072,row=pixel/64,col=pixel%64;
    float4 q=float4(qs[4*e],qs[4*e+1],qs[4*e+2],qs[4*e+3]);
    float w=q.x,x=q.y,y=q.z,z=q.w;
    float3 r0=float3(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w));
    float3 r1=float3(2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w));
    float3 r2=float3(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));
    float3 off=float3(.035f,0,.012f);
    float3 o=load3(positions+3*e)+float3(dot(r0,off),dot(r1,off),dot(r2,off));
    float fy=tan(1.099557429f/2);
    float3 b=float3(1,-(2*(col+.5f)/64-1)*fy*64/48,-(2*(row+.5f)/48-1)*fy);
    float3 d=normalize(float3(dot(r0,b),dot(r1,b),dot(r2,b)));
    float lo=0,hi=8,best=8; if(interval(o,d,room,lo,hi)) best=hi;
    bool obstacle=false; int winner=-1;
    for(int j=0;j<nb;j++) {
        lo=0;hi=8;
        if(interval(o,d,boxes+6*j,lo,hi) && lo<best) {best=lo;obstacle=true;}
    }
    for(int j=0;j<np;j++) {
        device const float* p=panels+14*j;
        float3 u=load3(p+3),v=load3(p+6),n=cross(u,v);
        float denom=dot(d,n); if(denom>=-1e-6f) continue;
        float t=dot(load3(p)-o,n)/denom;
        if(t<=0 || t>=best || t>8) continue;
        float3 h=o+t*d-load3(p);
        if(abs(dot(h,u))<=p[9] && abs(dot(h,v))<=p[10]) {best=t;winner=j;}
    }
    depth[idx]=best;
    if(winner>=0) for(int k=0;k<3;k++) rgb[idx*3+k]=uchar(panels[14*winner+11+k]);
    else {uchar c=obstacle?65:100+20*((row/8+col/8)%2);for(int k=0;k<3;k++) rgb[idx*3+k]=c;}
}

#include <metal_stdlib>
using namespace metal;

// Lens response operates on the actual illuminated, aerosol-attenuated camera RGB.
// Bright-source bloom and streaks are image formation effects, never observer overlays.
kernel void optics(device const uchar* source,device uchar* output,
    constant int& width,constant int& height,constant int& frame,
    uint idx [[thread_position_in_grid]]) {
    int x=idx%width,y=idx/width;
    if(y>=height)return;
    float3 c=float3(source[3*idx],source[3*idx+1],source[3*idx+2])/255.f;
    float3 bloom=0;
    for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++) {
        int xx=clamp(x+dx*2,0,width-1),yy=clamp(y+dy*2,0,height-1),j=(yy*width+xx)*3;
        float3 v=float3(source[j],source[j+1],source[j+2])/255.f;
        bloom+=max(v-.78f,0.f)/(1.f+dx*dx+dy*dy)*.12f;
    }
    float3 streak=0;
    for(int step=-5;step<=5;step++) {
        int xx=clamp(x+step*4,0,width-1),j=(y*width+xx)*3;
        float3 v=float3(source[j],source[j+1],source[j+2])/255.f;
        streak+=max(v-.9f,0.f)*(.07f/(1+abs(step)));
    }
    float2 uv=(float2(x+.5f,y+.5f)/float2(width,height)-.5f)*2;
    float vignette=1-.10f*dot(uv,uv);
    uint h=idx*747796405u+uint(frame)*2891336453u+277803737u;
    h=(h^(h>>16))*2246822519u;
    float noise=(float(h&65535)/65535.f-.5f)*.007f;
    c=(c+bloom+streak)*vignette+noise;
    for(int k=0;k<3;k++)output[3*idx+k]=uchar(clamp(c[k],0.f,1.f)*255.f);
}

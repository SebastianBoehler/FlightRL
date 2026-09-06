/* Analytic tree/rock surfaces inside conservative obstacle boxes. */
static int forest_hit(const float *o,const float *d,const float *b,float *near) {
    float c[3],s[3],v[3],w[3];
    int trunk=b[4]<.01f && b[5]>3;
    for(int k=0;k<3;k++){c[k]=(b[2*k]+b[2*k+1])*.5f;s[k]=(b[2*k+1]-b[2*k])*.5f;v[k]=(o[k]-c[k])/s[k];w[k]=d[k]/s[k];}
    if(trunk){v[2]=0;w[2]=0;}
    float a=dot(w,w), bb=dot(v,w), cc=dot(v,v)-1,disc=bb*bb-a*cc;
    if(a<1e-12f || disc<0)return 0;
    float lo=(-bb-sqrtf(disc))/a,hi=(-bb+sqrtf(disc))/a;
    if(trunk){
        if(fabsf(d[2])<1e-8f){if(o[2]<b[4] || o[2]>b[5])return 0;}
        else {float z0=(b[4]-o[2])/d[2],z1=(b[5]-o[2])/d[2];lo=fmaxf(lo,fminf(z0,z1));hi=fminf(hi,fmaxf(z0,z1));}
    }
    if(hi<0 || lo>hi)return 0;
    *near=fmaxf(0,lo);return *near<8;
}

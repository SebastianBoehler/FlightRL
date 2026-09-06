/* Camera-only procedural materials, direct illumination and glossy reflection.
 * Authored appearance model: no claim of calibrated radiometry or path tracing. */
static float clamp01(float x) { return fminf(1,fmaxf(0,x)); }
static int plant_window(const float *h,const float *windows,int count) {
    for(int i=0;i<count;i++) {
        const float *b=windows+i*6;
        if(h[0]>=b[0]-.01f && h[0]<=b[1]+.01f && h[1]>=b[2]-.01f && h[1]<=b[3]+.01f && h[2]>=b[4]-.01f && h[2]<=b[5]+.01f)return 1;
    }
    return 0;
}
static float plant_shadow(const float *h,const float *n,const float *ld,float length,
    const float *boxes,int count,int self) {
    float o[3]={h[0]+n[0]*.004f,h[1]+n[1]*.004f,h[2]+n[2]*.004f};
    for(int j=0;j<count;j++) {
        float near=.005f,far=length;
        if(j!=self && box_interval(o,ld,boxes+6*j,0,&near,&far))return 0;
    }
    return 1;
}
static void industrial_surface(const float *o,const float *d,float distance,
    const float *room,const float *boxes,int box_count,int box_index,
    int panel,const float *panels,uint8_t *rgb,const float *settings,const float *lights,int light_count,const float *windows,int window_count) {
    float h[3]={o[0]+d[0]*distance,o[1]+d[1]*distance,o[2]+d[2]*distance};
    const float *bounds=box_index>=0 ? boxes+6*box_index : room;
    float n[3]={0,0,0},closest=INFINITY;
    for(int k=0;k<3;k++) for(int side=0;side<2;side++) {
        float error=fabsf(h[k]-bounds[k*2+side]);
        if(error<closest){closest=error;n[0]=n[1]=n[2]=0;n[k]=(box_index>=0 ? 1:-1)*(side?1:-1);}
    }
    float base[3]={settings[1],settings[2],settings[3]},rough=settings[18],metal=settings[19];
    int machinery=box_index>=0 && bounds[5]-bounds[4]<2.9f;
    float u=fabsf(n[0])>.5f?h[1]:h[0];
    if(box_index>=0) {
        for(int k=0;k<3;k++)base[k]=settings[(machinery?4:1)+k];
        if(machinery) {
            rough=settings[7];metal=settings[8];
            float z=h[2]-bounds[4],center=fabsf(n[0])>.5f?(bounds[2]+bounds[3])*.5f:(bounds[0]+bounds[1])*.5f;
            float across=u-center;
            if(z<.07f || bounds[5]-h[2]<.05f){base[0]=base[1]=base[2]=165;rough=.22f;metal=.85f;}
            if(z<.8f && fmodf(z*18,1)<.28f)for(int k=0;k<3;k++)base[k]*=.35f;
            float gauge=across*across+(z-1.35f)*(z-1.35f);
            if(gauge<.0225f){base[0]=base[1]=base[2]=gauge>.016f?180:210;rough=.45f;}
            if(gauge<.014f && fabsf((z-1.35f)-across*.7f)<.012f)for(int k=0;k<3;k++)base[k]=25;
            if(fabsf(across)<.19f && z>.9f && z<1.05f){base[0]=20;base[1]=42;base[2]=48;}
            if(fabsf(across)<.15f && z>.93f && z<.98f && fmodf(u*75,1)>.3f){base[0]=92;base[1]=190;base[2]=158;}
            if(bounds[4]>2){base[0]=150;base[1]=157;base[2]=158;rough=.2f;metal=.9f;}
        }
    }
    if(box_index<0 && fabsf(n[2])>.5f) {
        float seams=fminf(fmodf(fabsf(h[0])*1.2f,1),fmodf(fabsf(h[1])*1.2f,1));
        for(int k=0;k<3;k++)base[k]=settings[13+k];
        if(seams<.015f)for(int k=0;k<3;k++)base[k]*=.65f;
        float wet=sinf(h[0]*1.7f+h[1]*.8f)*sinf(h[1]*2.1f);
        rough=wet>.35f?settings[16]*.25f:settings[16];metal=settings[17];
        if(h[2]<.1f && fabsf(fmodf(h[1]+3,2.7f)-.12f)<.035f){base[0]=225;base[1]=179;base[2]=59;}
    }
    if(settings[20]==1 && machinery) {
        float slot=fmodf(fabsf(h[2])*14,1);
        for(int k=0;k<3;k++)base[k]=settings[4+k]*(slot<.12f?.35f:1);
        if(fmodf(fabsf(u)*35,1)<.12f && slot>.3f && slot<.5f){base[0]=48;base[1]=180;base[2]=155;}
        rough=.4f;metal=.65f;
    }
    if(settings[20]==2 && panel<0) {
        rough=.95f;metal=0;
        if(box_index>=0) {
            int trunk=bounds[4]<.01f && bounds[5]>3;
            float norm=0;
            for(int k=0;k<3;k++) {float half=(bounds[2*k+1]-bounds[2*k])*.5f;n[k]=(h[k]-(bounds[2*k]+bounds[2*k+1])*.5f)/(half*half);if(trunk && k==2)n[k]=0;norm+=n[k]*n[k];}
            norm=sqrtf(norm);for(int k=0;k<3;k++)n[k]/=fmaxf(norm,1e-6f);
            float texture=.75f+.25f*sinf(h[0]*53+h[1]*37+h[2]*(trunk?2:43));
            float tint[3]={trunk?91:bounds[4]>1?48:103,trunk?65:bounds[4]>1?98:108,trunk?40:bounds[4]>1?41:91};
            for(int k=0;k<3;k++)base[k]=tint[k]*texture;
        } else if(h[2]>.05f) {
            float sky=clamp01(d[2]*.5f+.5f);rgb[0]=(uint8_t)(150-55*sky);rgb[1]=(uint8_t)(185-30*sky);rgb[2]=(uint8_t)(195+35*sky);return;
        } else {
            float litter=.75f+.25f*sinf(h[0]*45)*sinf(h[1]*39);
            for(int k=0;k<3;k++)base[k]=settings[13+k]*litter;
        }
    }
    if(panel>=0) {
        const float *p=panels+14*panel,*a=p+3,*b=p+6;
        n[0]=a[1]*b[2]-a[2]*b[1];n[1]=a[2]*b[0]-a[0]*b[2];n[2]=a[0]*b[1]-a[1]*b[0];
        for(int k=0;k<3;k++)base[k]=p[11+k];rough=.75f;metal=0;
    }
    if(box_index<0 && panel<0 && plant_window(h,windows,window_count)) {
        float bars=fmodf(h[0]-room[0],.8f)<.035f || fabsf(h[2]-2.65f)<.025f;
        rgb[0]=bars?64:236;rgb[1]=bars?73:245;rgb[2]=bars?79:255;return;
    }
    float color[3];for(int k=0;k<3;k++)color[k]=base[k]*settings[0];
    for(int light=0;light<=light_count;light++) {
        float ld[3],power=1;
        if(light<light_count){for(int k=0;k<3;k++)ld[k]=lights[7*light+k]-h[k];}
        else {for(int k=0;k<3;k++)ld[k]=settings[9+k];}
        float length=sqrtf(dot(ld,ld));for(int k=0;k<3;k++)ld[k]/=length;
        if(light<light_count)power=lights[7*light+6]/(1+.35f*length*length);
        else {
            float near=0,far=20,hit[3];
            box_interval(h,ld,room,0,&near,&far);
            for(int k=0;k<3;k++)hit[k]=h[k]+ld[k]*far;
            power=(settings[20]==2 || plant_window(hit,windows,window_count))?settings[12]:0;length=far;
        }
        if(power==0)continue;
        power*=plant_shadow(h,n,ld,length,boxes,box_count,box_index);
        if(power==0)continue;
        float lambert=fmaxf(0,dot(n,ld));
        float half[3]={ld[0]-d[0],ld[1]-d[1],ld[2]-d[2]};
        float norm=sqrtf(dot(half,half));for(int k=0;k<3;k++)half[k]/=fmaxf(norm,1e-6f);
        float fresnel=.04f+.5f*metal+.25f*powf(1-clamp01(-dot(n,d)),5);
        float spec=powf(fmaxf(0,dot(n,half)),2+80*(1-rough))*fresnel*180;
        for(int k=0;k<3;k++)color[k]+=power*(base[k]*lambert*.65f+spec)*(light==light_count?1:lights[7*light+3+k]);
    }
    float grain=1.5f*sinf(h[0]*173+h[1]*127+h[2]*191);
    for(int k=0;k<3;k++)rgb[k]=(uint8_t)fminf(255,fmaxf(0,color[k]+grain));
}

import * as T from 'three/webgpu';
interface Request {
  scene:T.Scene; position:T.Vector3; quaternion:T.Quaternion;
  connected:()=>boolean; stillCurrent:()=>boolean;
  publish:(canvas:HTMLCanvasElement)=>void;
}
/** Camera and overview use the same renderer and synchronous presentation cycle. */
export class ForestFeed {
  private camera=new T.PerspectiveCamera(63,4/3,.035,80);
  private pending:Request|null=null;
  private canvas=document.createElement('canvas');
  private frames=0;
  private reportAt=0;
  constructor(){this.camera.layers.set(1);this.canvas.width=512;this.canvas.height=384;}
  async start() {}
  reset(){this.pending=null;this.frames=0;this.reportAt=performance.now();}
  async render(scene:T.Scene,pose:T.Camera,connected:()=>boolean,stillCurrent:()=>boolean,publish:(canvas:HTMLCanvasElement)=>void){
    this.pending={scene,position:pose.position.clone(),quaternion:pose.quaternion.clone(),connected,stillCurrent,publish};
  }
  flush(renderer:T.WebGPURenderer){
    const request=this.pending;this.pending=null;
    if(!request || !request.stillCurrent())return;
    this.camera.position.copy(request.position);this.camera.quaternion.copy(request.quaternion);
    const size=renderer.getSize(new T.Vector2());
    const width=Math.min(512,size.x,size.y*4/3),height=width*3/4;
    const viewport=renderer.getViewport(new T.Vector4()),scissor=renderer.getScissor(new T.Vector4());
    const scissorTest=renderer.getScissorTest();
    try{
      renderer.setViewport(0,0,width,height);renderer.setScissor(0,0,width,height);renderer.setScissorTest(true);
      renderer.render(request.scene,this.camera);
      const ratio=renderer.getPixelRatio();
      // Capture in the same task as submission, before canvas presentation.
      this.canvas.getContext('2d')!.drawImage(renderer.domElement,0,0,width*ratio,height*ratio,0,0,512,384);
      for(const id of request.connected()?['camera','operator-camera']:['camera']){
        const target=document.getElementById(id) as HTMLCanvasElement;
        if(target.width!==512 || target.height!==384){target.width=512;target.height=384;}
        target.getContext('2d')!.drawImage(this.canvas,0,0);
      }
      request.publish(this.canvas);
      this.frames++;
      const now=performance.now();
      if(now-this.reportAt>=1000){
        document.getElementById('camera-freshness')!.textContent=`Shared render cycle · ${(1000*this.frames/(now-this.reportAt)).toFixed(1)} camera fps`;
        this.frames=0;this.reportAt=now;
      }
    }finally{renderer.setViewport(viewport);renderer.setScissor(scissor);renderer.setScissorTest(scissorTest);}
  }
}

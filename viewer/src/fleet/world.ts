import * as T from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { disposeScene } from '../dispose-scene';
import {habitat} from './habitat';
import type { FleetFrame, FleetReplay } from './types';
export const colors = [0x50daca, 0xffb45c, 0x96a7ff];
export class FleetWorld {
  renderer = new T.WebGPURenderer({antialias:true});
  scene = new T.Scene();
  observer = new T.PerspectiveCamera(55, 1, .03, 100);
  cameras = colors.map(() => new T.PerspectiveCamera(63, 4/3, .035, 100));
  drones: T.Group[] = [];
  targets: T.Mesh[] = [];
  trails: T.Line[] = [];
  taskMarkers: T.Mesh<T.CylinderGeometry,T.MeshBasicMaterial>[] = [];
  controls: OrbitControls;
  forest: ReturnType<typeof habitat>;
  follow = false;
  selected = 0;
  constructor(private replay: FleetReplay, private panes: HTMLElement[]) {
    T.Object3D.DEFAULT_UP.set(0,0,1);
    this.observer.up.set(0,0,1);
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.domElement.id = 'fleet-renderer';
    panes[0].prepend(this.renderer.domElement);
    this.forest = habitat(this.scene,replay);
    this.scene.traverse(o => o.userData.cameraOnly?o.layers.set(1):o.layers.enable(1));
    this.controls = new OrbitControls(this.observer, panes[0]);
    this.controls.enableDamping = true;
    this.controls.target.set(2,-1,3);
    this.overview();
    replay.tasks?.forEach((p,i)=>{
      const marker=new T.Mesh(new T.CylinderGeometry(.19,.19,.08,24),new T.MeshBasicMaterial({color:0xffd88a}));
      marker.rotation.x=Math.PI/2;marker.position.set(p[0],p[1],.1);this.scene.add(marker);this.taskMarkers.push(marker);
      const label=document.createElement('span');label.className='task-label';label.id=`task-label-${i}`;label.textContent=`T${i+1}`;panes[0].append(label);
    });
    colors.forEach((color,i) => {
      const drone = new T.Group();
      const body = new T.Mesh(new T.BoxGeometry(.12,.09,.05),new T.MeshStandardMaterial({color}));
      drone.add(body);
      for(const x of [-.065,.065]) for(const y of [-.075,.075]) {
        const rotor = new T.Mesh(new T.CylinderGeometry(.035,.035,.006,20),new T.MeshStandardMaterial({color:0x333940}));
        rotor.rotation.x = Math.PI/2; rotor.position.set(x,y,0); drone.add(rotor);
      }
      drone.traverse(o => o.layers.enable(1));
      const halo = new T.Mesh(new T.RingGeometry(.26,.29,32),new T.MeshBasicMaterial({color,side:T.DoubleSide}));
      drone.add(halo); this.scene.add(drone); this.drones.push(drone);
      const target = new T.Mesh(new T.OctahedronGeometry(.18),new T.MeshBasicMaterial({color,wireframe:true}));
      this.scene.add(target); this.targets.push(target);
      const geometry = new T.BufferGeometry().setFromPoints(replay.records.map(f => new T.Vector3(...f.positions[i] as [number,number,number])));
      const trail = new T.Line(geometry,new T.LineBasicMaterial({color}));
      this.scene.add(trail); this.trails.push(trail);
      this.cameras[i].layers.set(1);
    });
  }
  dispose() { this.renderer.setAnimationLoop(null); this.controls.dispose(); disposeScene(this.scene); this.renderer.dispose(); }
  overview() { this.follow = false; this.observer.position.set(12,-15,12); this.controls.target.set(2,-1,3); if(this.replay.sensor_atlas){this.observer.position.set(10,-9,7);this.controls.target.set(2,0,1.7);} this.controls.update(); }
  update(frame: FleetFrame, index: number) {
    this.forest.update(frame.time_s, [0,0,0]);
    this.taskMarkers.forEach((marker,i)=>{
      const done=frame.task_done?.[i],found=frame.task_found?.[i];marker.material.color.set(done?0x59df90:found?0x50daca:0xffd88a);
      document.getElementById(`task-label-${i}`)!.textContent=`${done?'✓ ':found?'Found ':''}${frame.task_found?'S':'T'}${i+1}`;
    });
    frame.positions.forEach((p,i) => {
      const drone=this.drones[i], q=frame.quaternions[i];
      drone.position.set(p[0],p[1],p[2]); drone.quaternion.set(q[1],q[2],q[3],q[0]);
      const forward=new T.Vector3(1,0,0).applyQuaternion(drone.quaternion);
      const left=new T.Vector3(0,1,0).applyQuaternion(drone.quaternion);
      const up=new T.Vector3(0,0,1).applyQuaternion(drone.quaternion);
      this.cameras[i].position.copy(drone.position).addScaledVector(forward,.12);
      this.cameras[i].quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(left.negate(),up,forward.negate()));
      this.targets[i].position.fromArray(frame.goals[i]);
      this.trails[i].geometry.setDrawRange(0,index+1);
    });
    if(this.follow) {
      const p=this.drones[this.selected].position;
      this.observer.position.copy(p).add(new T.Vector3(-3,-4,2)); this.controls.target.copy(p);
    }
  }
  async draw() {
    const host = this.panes[0], width = host.clientWidth, height = host.clientHeight;
    this.controls.update();
    const views = [{host, camera: this.observer}, ...this.panes.slice(1).map((host, i) => ({host, camera: this.cameras[i]}))];
    for (let i = this.replay.sensor_atlas ? 0 : views.length - 1; i >= 0; i--) {
      const entry = views[i], w = i === 0 ? width : 256, h = i === 0 ? height : 192;
      if (!w || !h || (i > 0 && !entry.host.clientWidth)) continue;
      this.renderer.setSize(w, h, false);
      entry.camera.aspect = w / h; entry.camera.updateProjectionMatrix();
      if (i > 0) this.drones[i - 1].visible = false;
      this.renderer.render(this.scene, entry.camera);
      if (i > 0) {
        (entry.host as HTMLCanvasElement).getContext('2d')!.drawImage(this.renderer.domElement, 0, 0, 256, 192);
        this.drones[i - 1].visible = true;
      }
    }
    const pane=this.panes[0].getBoundingClientRect();
    this.drones.forEach((drone,i)=>{
      const p=drone.position.clone().project(this.observer), label=document.getElementById(`drone-label-${i}`)!;
      label.hidden=Math.abs(p.x)>1 || Math.abs(p.y)>1 || p.z<0 || p.z>1;
      label.style.left=`${(p.x+1)*pane.width/2}px`;label.style.top=`${(1-p.y)*pane.height/2}px`;
    });
    this.taskMarkers.forEach((marker,i)=>{
      const p=marker.position.clone().project(this.observer), label=document.getElementById(`task-label-${i}`)!;
      label.hidden=Math.abs(p.x)>1 || Math.abs(p.y)>1 || p.z<0 || p.z>1;
      label.style.left=`${(p.x+1)*pane.width/2}px`;label.style.top=`${(1-p.y)*pane.height/2}px`;
    });
    await this.renderer.waitForGPU();
  }
}

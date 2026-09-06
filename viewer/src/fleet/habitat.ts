import * as T from 'three/webgpu';
import {buildForest} from '../forest/trees';
import {forestGround} from '../forest/ground';
import {sunrise} from '../forest/sunrise';
import {plantDetails} from '../plant-details';
import type {FleetReplay} from './types';

export function habitat(scene:T.Scene,replay:FleetReplay){
  if(replay.provenance.family==='forest'){
    sunrise(scene);const forest=buildForest([0,0,0]);forestGround(forest.root);scene.add(forest.root);
    return forest;
  }
  if(!['utility_plant','flight_course','camera_control'].includes(replay.provenance.family))throw Error('Unsupported fleet habitat');
  const root=new T.Group();scene.add(root);scene.background=new T.Color(0x17232d);
  scene.add(new T.HemisphereLight(0xdceeff,0x4f575a,2.5));
  const light=new T.DirectionalLight(0xe1efff,3);light.position.set(2,-1,6);light.castShadow=true;light.shadow.mapSize.set(2048,2048);scene.add(light);
  const box=(bounds:number[],color:number,metalness=.15)=>{
    const size=[bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4]];
    const mesh=new T.Mesh(new T.BoxGeometry(...size as [number,number,number]),new T.MeshStandardMaterial({color,metalness,roughness:.55}));
    mesh.position.set((bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2,(bounds[4]+bounds[5])/2);mesh.castShadow=true;mesh.receiveShadow=true;root.add(mesh);return mesh;
  };
  for(const p of replay.scene.panels??[]){
    const panel=new T.Mesh(new T.PlaneGeometry(p[9]*2,p[10]*2),new T.MeshBasicMaterial({color:new T.Color(p[11]/255,p[12]/255,p[13]/255),side:T.DoubleSide}));
    const u=new T.Vector3(...p.slice(3,6) as [number,number,number]),v=new T.Vector3(...p.slice(6,9) as [number,number,number]);
    panel.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(u,v,new T.Vector3().crossVectors(u,v)));panel.position.fromArray(p);root.add(panel);
  }
  const b=replay.scene.room;
  box([b[0],b[1],b[2],b[3],-.12,0],0x65737b);
  for(const obstacle of replay.scene.boxes)box(obstacle,obstacle[4]>2?0x81999e:0x33444e,.5);
  if(replay.provenance.family==='utility_plant')plantDetails(root,replay.scene.boxes);
  for(let x=b[0]+.2;x<b[1];x+=.8){
    box([x,x+.015,b[2],b[3],.001,.004],0x9aa8ad);
  }
  for(let x=b[0]+.5;x<b[1];x+=2){
    const lamp=box([x,x+1,b[2]+.2,b[2]+.3,b[5]-.1,b[5]-.06],0xffffff);
    (lamp.material as T.MeshStandardMaterial).emissive.set(0xc7eaff);(lamp.material as T.MeshStandardMaterial).emissiveIntensity=2;
  }
  // Enclosure exists in camera views; near walls are cut away in the overview.
  for(const bounds of [[b[0]-.1,b[0],b[2],b[3],0,b[5]],[b[1],b[1]+.1,b[2],b[3],0,b[5]],[b[0],b[1],b[2]-.1,b[2],0,b[5]],[b[0],b[1],b[3],b[3]+.1,0,b[5]]]){
    const wall=box(bounds,0x7f8c90);wall.userData.cameraOnly=true;
  }
  return {root,update:(_time:number,_wind:number[])=>{}};
}

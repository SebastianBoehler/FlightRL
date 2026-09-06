import type {FleetReplay} from './types';

/** Predecode exact actor images before allowing the replay clock to advance. */
export async function sensorRecording(replay:FleetReplay){
  if(!replay.sensor_atlas)return null;
  const response=await fetch(replay.sensor_atlas);
  if(!response.ok)throw Error(`Sensor recording HTTP ${response.status}`);
  const atlas=await createImageBitmap(await response.blob());
  if(atlas.width!==64*3 || atlas.height!==48*replay.records.length)throw Error('Sensor atlas does not match replay');
  const contexts=Array.from({length:3},(_,i)=>{
    const canvas=document.createElement('canvas');canvas.width=64;canvas.height=48;
    canvas.className='recorded-sensor';canvas.setAttribute('aria-label',`Drone ${i+1} recorded policy RGB`);
    document.getElementById(`camera-${i}`)!.append(canvas);
    const context=canvas.getContext('2d');if(!context)throw Error('2D sensor canvas unavailable');
    return context;
  });
  return {draw(index:number){contexts.forEach((ctx,i)=>ctx.drawImage(atlas,i*64,index*48,64,48,0,0,64,48));}};
}

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as T from 'three/webgpu';
import { disposeScene } from '../src/dispose-scene.ts';

test('scene replacement releases shared geometry, textures, instances and shadows', () => {
  const root = new T.Group();
  const geometry = new T.BoxGeometry();
  const texture = new T.Texture();
  const material = new T.MeshStandardMaterial({ map: texture, bumpMap: texture });
  const instances = new T.InstancedMesh(geometry, material, 2);
  const light = new T.DirectionalLight();
  light.shadow.map = new T.RenderTarget(16, 16);
  root.add(instances, new T.Mesh(geometry, material), light);
  const disposed = [];
  for (const [name, object] of Object.entries({ geometry, texture, material, instances, shadow: light.shadow.map }))
    object.addEventListener('dispose', () => disposed.push(name));
  disposeScene(root);
  assert.deepEqual(disposed.sort(), ['geometry', 'instances', 'material', 'shadow', 'texture']);
});

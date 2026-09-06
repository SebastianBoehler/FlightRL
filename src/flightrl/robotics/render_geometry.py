"""Serialize compiled mesh vertices and physical transforms without CAD guesses."""

import mujoco as mj


def geometry_description(model):
    types = {
        mj.mjtGeom.mjGEOM_BOX: "box",
        mj.mjtGeom.mjGEOM_SPHERE: "sphere",
        mj.mjtGeom.mjGEOM_CYLINDER: "cylinder",
        mj.mjtGeom.mjGEOM_CAPSULE: "capsule",
        mj.mjtGeom.mjGEOM_MESH: "mesh",
    }
    meshes = {}
    geoms = []
    for i in range(model.ngeom):
        kind = int(model.geom_type[i])
        if kind not in types:
            raise ValueError(f"Unsupported render geometry type {kind} at geom {i}")
        # Menagerie uses group 2 for visual meshes and group 3 for collision proxies.
        if model.geom_group[i] == 3:
            continue
        mesh_id = int(model.geom_dataid[i]) if types[kind] == "mesh" else None
        if mesh_id is not None and mesh_id not in meshes:
            v, f = model.mesh_vertadr[mesh_id], model.mesh_faceadr[mesh_id]
            meshes[mesh_id] = dict(
                vertices=model.mesh_vert[v : v + model.mesh_vertnum[mesh_id]]
                .ravel()
                .tolist(),
                indices=model.mesh_face[f : f + model.mesh_facenum[mesh_id]]
                .ravel()
                .tolist(),
            )
        material = int(model.geom_matid[i])
        color = model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]
        geoms.append(
            dict(
                name=mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}",
                body=int(model.geom_bodyid[i]),
                type=types[kind],
                mesh=mesh_id,
                size=model.geom_size[i].tolist(),
                position=model.geom_pos[i].tolist(),
                quaternion=model.geom_quat[i][[1, 2, 3, 0]].tolist(),
                color=color.tolist(),
            )
        )
    return dict(geometries=geoms, meshes=meshes)

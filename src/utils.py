import numpy as np
import os
import yaml

# Generate marker
def create_marker(viewer, mujoco):
  scene = viewer.user_scn
  num_geoms = scene.ngeom
  geom = scene.geoms[num_geoms]
  mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.array([0.05, 0.05, 0.05]),  # size
      np.array([0, 0, 0], dtype=np.float64),  # pos
      np.eye(3).flatten(),  # rotation matrix (identity)
      np.array([1, 0, 0.9, 1], dtype=np.float32)  # rgba
    )
  scene.ngeom += 1
  return geom

# Grab yaml content as python list
def get_yaml_content(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = yaml.safe_load(f)
            return content
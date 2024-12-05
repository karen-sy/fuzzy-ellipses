import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import utils

import sys
import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# TODO make fuzzy-ellipses an installable package
root_path = "/home/karen/fuzzy-ellipses"
sys.path.append(root_path)

ycb_dir = f"{root_path}/assets/bop/ycbv/train_real"
ycb_dir

all_data = utils.get_ycbv_data(ycb_dir, 1, [1], fields=[])
fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
# cam_pose = all_data[0]["camera_pose"]  # NOT identity but obj pose is in world frame already

# mesh data
object_index = 1
mesh = utils.get_ycb_mesh(ycb_dir, all_data[0]["object_types"][object_index])
mesh_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
object_pose = all_data[0]["object_poses"][object_index]  # pose in world frame
_mesh_vtx = mesh.vertices  # untransformed vertices

# image data
ycb_rgb = all_data[0]["rgb"]
ycb_depth = all_data[0]["depth"]
height, width = ycb_rgb.shape[:2]
del all_data

# GT viz: transform and project mesh to rgbd
mesh_vtx = utils.transform_points(object_pose, _mesh_vtx)

x = np.array(fy * mesh_vtx[:, 1] / mesh_vtx[:, 2] + cy, dtype=np.int32)
y = np.array(fx * mesh_vtx[:, 0] / mesh_vtx[:, 2] + cx, dtype=np.int32)
print(f"2D x range = ({np.min(x)}, {np.max(x)})")
print(f"2D y range = ({np.min(y)}, {np.max(y)})")
pixels = np.stack([x, y], axis=1)
rgb = np.zeros((height, width, 3), dtype=np.uint8)
depth = np.zeros((height, width, 1))

rgb[pixels[:, 0], pixels[:, 1], :] = mesh_colors * 255
depth[pixels[:, 0], pixels[:, 1], 0] = mesh_vtx[:, 2]

###
exp_pixels = jnp.array()  # paste value here

exp_depth = np.zeros((height, width, 1))
exp_depth[exp_pixels[:, 1], exp_pixels[:, 0], 0] = (
    1.0  # x, y flipped for python indexing
)

# image data
image_size = (height, width)
color, target_depth = rgb, depth[..., 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(
    jnp.where(depth[..., 0] > 0, depth[..., 0], 0).reshape(image_size), cmap="Greys"
)

axes[1].imshow(
    jnp.where(exp_depth[..., 0] > 0, exp_depth[..., 0], 0).reshape(image_size),
    cmap="Greys",
)
blended_image = jnp.where(exp_depth[..., 0] > 0, depth[..., 0], 0).reshape(
    image_size
) - jnp.where(exp_depth[..., 0] > 0, exp_depth[..., 0], 0).reshape(image_size)

axes[2].imshow(blended_image, cmap="Greys")

fig.tight_layout()


# save plot and binary data
folder_name = f"{width}_{height}_VIZ"
directory = f"{root_path}/data/{folder_name}"
if not os.path.exists(directory):
    os.makedirs(directory)
fig.savefig(f"{directory}/python_ref.png")

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import utils

import sys

import os
import baseline.fm_render as fm_render
from baseline.util_render import quat_to_rot


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# TODO make fuzzy-ellipses an installable package
root_path = "/home/karen/fuzzy-ellipses"
sys.path.append(root_path)

ycb_dir = f"{root_path}/assets/bop/ycbv/train_real"
ycb_dir


all_data = utils.get_ycbv_data(ycb_dir, 1, [1], fields=[])
fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
cam_pose = all_data[0]["camera_pose"]

# mesh data
object_index = 1
mesh = utils.get_ycb_mesh(ycb_dir, all_data[0]["object_types"][object_index])
mesh_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
object_pose = all_data[0]["object_poses"][object_index]
_mesh_vtx = mesh.vertices  # untransformed vertices

# image data
ycb_rgb = all_data[0]["rgb"]
ycb_depth = all_data[0]["depth"]
height, width = 480, 480

del all_data


# transform and project mesh to rgbd
mesh_vtx = utils.transform_points(object_pose, _mesh_vtx)

x = np.array(fy * mesh_vtx[:, 1] / mesh_vtx[:, 2] + cy, dtype=np.int32)
y = np.array(fx * mesh_vtx[:, 0] / mesh_vtx[:, 2] + cx, dtype=np.int32)
pixels = np.stack([x, y], axis=1)
rgb = np.zeros((height, width, 3), dtype=np.uint8)
depth = np.zeros((height, width, 1))

rgb[pixels[:, 0], pixels[:, 1], :] = mesh_colors * 255
depth[pixels[:, 0], pixels[:, 1], 0] = mesh_vtx[:, 2] * 1000


##### adapt setup to FMB repo boilerplate

id_pose = jnp.array([0, 0, 0, 1, 0, 0, 0])

# object data
gt_trans = id_pose[:3]
gt_quat = id_pose[3:]
shape_scale = 1.0

# image data
image_size = (height, width)
color, target_depth = rgb, depth[..., 0]


# volume usually False since color optimization implies surface samples
# And code defaults towards that sort of usage now
show_volume = False

beta2 = 21.4  # how did fmb choose this?
beta3 = 2.66

render = fm_render.render_func_quat
render_jit = jax.jit(fm_render.render_func_quat)


# port FMB optimization setup
NUM_MIXTURE = 150
rng = np.random.default_rng(1222)
pts = mesh_vtx[rng.integers(0, len(mesh_vtx), NUM_MIXTURE)]
weights_log = np.log(np.ones((NUM_MIXTURE,)) / NUM_MIXTURE)
mean = pts
cov = np.array([np.eye(3) for _ in range(NUM_MIXTURE)]) / 1e6
_inv_cov = np.linalg.inv(cov)
prec = np.linalg.cholesky(_inv_cov)

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
pixel_list = (
    (np.array(np.meshgrid(np.arange(width), np.arange(height), [0]))[:, :, :, 0])
    .reshape((3, -1))
    .T
)
camera_rays = (pixel_list - K[:, 2]) / np.diag(K)
camera_rays[:, -1] = 1
del pixel_list


# render!
est_depth_true, est_alpha_true, _, _ = render_jit(
    mean,
    prec,
    weights_log,
    camera_rays,
    gt_quat,
    gt_trans,
    beta2 / shape_scale,
    beta3,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
_est_depth_true = np.asarray(est_depth_true)  # copy
axes[0].imshow(_est_depth_true.reshape(image_size))
axes[0].set_title("depth @ gt pose")

_est_alpha_true = np.asarray(est_alpha_true)  # copy
axes[1].imshow(_est_alpha_true.reshape(image_size), cmap="Greys")
axes[1].set_title("alpha @ gt pose")

fig.tight_layout()


# save plot and binary data
directory = f"{root_path}/data"
if not os.path.exists(directory):
    os.makedirs(directory)
fig.savefig(f"{directory}/python_ref.png")

_est_depth_true.ravel().tofile(f"{directory}/zs.bin")
_est_alpha_true.ravel().tofile(f"{directory}/alphas.bin")
np.asarray(mean, dtype=np.float32).ravel().tofile(f"{directory}/means.bin")
np.asarray(prec, dtype=np.float32).ravel().tofile(f"{directory}/precs.bin")
np.asarray(weights_log, dtype=np.float32).ravel().tofile(f"{directory}/weights.bin")
np.asarray(camera_rays, dtype=np.float32).ravel().tofile(f"{directory}/camera_rays.bin")
np.asarray(quat_to_rot(gt_quat), dtype=np.float32).ravel().tofile(
    f"{directory}/camera_rot.bin"
)
np.asarray(gt_trans, dtype=np.float32).ravel().tofile(f"{directory}/camera_trans.bin")

# also save the transformed camera rays for reference
camera_rays_xfm = camera_rays @ quat_to_rot(gt_quat)  # (pixels, 3)
np.asarray(camera_rays_xfm, dtype=np.float32).ravel().tofile(
    f"{directory}/camera_rays_xfm.bin"
)

for var in [
    "zs",
    "alphas",
    "means",
    "precs",
    "weights",
    "camera_rays",
    "camera_rot",
    "camera_trans",
    "camera_rays_xfm",
]:
    arr = np.fromfile(f"{directory}/{var}.bin", dtype=np.float32)
    print(f"{var}[:3] in .bin: {arr.ravel()[:3]}; shape: {arr.shape}")

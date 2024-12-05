import math
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import utils
from time import time

import sys
from typing import NamedTuple
import os
import baseline.fm_render as fm_render
from baseline.util_render import quat_to_rot


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# TODO make fuzzy-ellipses an installable package
root_path = "/home/karen/fuzzy-ellipses"
sys.path.append(root_path)

ycb_dir = f"{root_path}/assets/bop/ycbv/train_real"
ycb_dir


class Intrinsics(NamedTuple):
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float


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

# intrinsics
intrinsics = Intrinsics(
    height=height,
    width=width,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    near=0.01,
    far=ycb_depth.max() * 1.5,
)

# GT viz: transform and project mesh to rgbd
mesh_vtx = utils.transform_points(object_pose, _mesh_vtx)
print(f"{mesh_vtx.shape} vertices in mesh")
print(f"{np.unique(mesh_vtx, axis=0).shape} unique vertices in mesh")

x = np.array(fy * mesh_vtx[:, 1] / mesh_vtx[:, 2] + cy, dtype=np.int32)
y = np.array(fx * mesh_vtx[:, 0] / mesh_vtx[:, 2] + cx, dtype=np.int32)
print(f"2D x range = ({np.min(x)}, {np.max(x)})")
print(f"2D y range = ({np.min(y)}, {np.max(y)})")
pixels = np.stack([x, y], axis=1)
rgb = np.zeros((height, width, 3), dtype=np.uint8)
depth = np.zeros((height, width, 1))

rgb[pixels[:, 0], pixels[:, 1], :] = mesh_colors * 255
depth[pixels[:, 0], pixels[:, 1], 0] = mesh_vtx[:, 2]


##### adapt setup to FMB repo boilerplate

id_pose = np.array([0, 0, 0, 1, 0, 0, 0])
gt_cam_trans = id_pose[:3]
gt_cam_quat = id_pose[3:]
gt_cam_rot_3x3 = quat_to_rot(gt_cam_quat)

# cam pose data
gt_cam_4x4 = np.zeros((4, 4))
gt_cam_4x4[-1, -1] = 1
gt_cam_4x4[:3, :3] = gt_cam_rot_3x3[:3, :3]
gt_cam_4x4[:3, -1] = gt_cam_trans

shape_scale = 1.0
fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
tan_fovx = math.tan(fovX)
tan_fovy = math.tan(fovY)
_proj_matrix = utils.getProjectionMatrixJax(
    width,
    height,
    intrinsics.fx,
    intrinsics.fy,
    intrinsics.cx,
    intrinsics.cy,
    intrinsics.near,
    intrinsics.far,
)
view_matrix = jnp.transpose(jnp.linalg.inv(gt_cam_4x4))
proj_matrix = view_matrix @ _proj_matrix
print("view, proj:", view_matrix, proj_matrix)

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
for NUM_MIXTURE in [150, 500, 1500, 2500, 3500]:
    DO_RENDER = NUM_MIXTURE <= 5000
    print(f"\n==========NUM_MIXTURE = {NUM_MIXTURE}==========")

    # save plot and binary data
    folder_name = f"{width}_{height}_{NUM_MIXTURE}"
    directory = f"{root_path}/data/{folder_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    rng = np.random.default_rng(1222)
    pts = mesh_vtx[rng.integers(0, len(mesh_vtx), NUM_MIXTURE)]
    gmm_init_scale = 80
    weights_log = np.log(np.ones((NUM_MIXTURE,)) / NUM_MIXTURE) + np.log(gmm_init_scale)
    mean = pts

    # Sigma = RLR^T where L diagonal elements are eigenvalues = RSS^TR^T where s = sqrt(eigenvalue)
    gaussian_quat = np.array(
        [id_pose[3:] for _ in range(NUM_MIXTURE)]
    )  # identity rotation
    _rot = jax.lax.map(quat_to_rot, gaussian_quat, batch_size=256)  # identity rotation
    scale = np.array([1 / jnp.sqrt(5e3) * np.eye(3) for _ in range(NUM_MIXTURE)])

    rs = jnp.einsum(
        "aij,ajk->aik", _rot, scale
    )  # cov = (rot @ scale) @ (rot @ scale).T
    cov = jnp.einsum("aij,ajk->aik", rs, rs.transpose(0, 2, 1))
    prec = np.linalg.inv(cov)
    prec_sqrt = np.linalg.cholesky(
        prec
    )  # A = L L^T for real A, L lower triangular with positive diagonal

    # camera
    fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
    fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
    tan_fovx = math.tan(fovX)
    tan_fovy = math.tan(fovY)

    K = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ]
    )
    pixel_list = (
        (np.array(np.meshgrid(np.arange(width), np.arange(height), [0]))[:, :, :, 0])
        .reshape((3, -1))
        .T
    )
    camera_rays = (pixel_list - K[:, 2]) / np.diag(K)
    camera_rays[:, -1] = 1
    del pixel_list

    # render!
    _est_depth_true = jnp.zeros((height * width))
    _est_alpha_true = jnp.zeros((height * width))
    if DO_RENDER:
        start = time()
        est_depth_true, est_alpha_true, _, _ = render(
            mean,
            prec_sqrt,
            weights_log,
            camera_rays,
            gt_cam_quat,
            gt_cam_trans,
            beta2 / shape_scale,
            beta3,
        )
        finish = time()
        time_elapsed = finish - start
        print(f"Render time: {time_elapsed:.4f}s = {1/time_elapsed}FPS")

        # print some stats
        print(f"z range = [{np.min(est_depth_true)}, {np.max(est_depth_true)}]")
        print(f"alpha range = [{np.min(est_alpha_true)}, {np.max(est_alpha_true)}]")

        # visualize
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        _est_depth_true = np.asarray(est_depth_true)  # copy
    axes[0].imshow(_est_depth_true.reshape(image_size))
    axes[0].set_title("depth @ gt pose")

    _est_alpha_true = np.asarray(est_alpha_true)  # copy
    axes[1].imshow(_est_alpha_true.reshape(image_size), cmap="Greys")
    axes[1].set_title("alpha @ gt pose")

    _est_alpha_true = np.asarray(est_alpha_true)  # copy
    axes[2].imshow(
        jnp.where(est_alpha_true > 0.5, _est_depth_true, jnp.nan).reshape(image_size)
    )
    axes[2].set_title("depth @ gt pose (alpha > 0.5)")

    axes[3].imshow(
        jnp.where(depth[..., 0] > 0, depth[..., 0], 0).reshape(image_size), cmap="Greys"
    )
    axes[3].set_title("gt depth")

    fig.tight_layout()

    fig.savefig(f"{directory}/python_ref.png")

    # save binary data into .bin
    np.asarray(_est_depth_true, dtype=np.float32).ravel().tofile(f"{directory}/zs.bin")
    np.asarray(_est_alpha_true, dtype=np.float32).ravel().tofile(
        f"{directory}/alphas.bin"
    )
    np.asarray(mean, dtype=np.float32).ravel().tofile(f"{directory}/means.bin")
    np.asarray(gaussian_quat, dtype=np.float32).ravel().tofile(f"{directory}/quats.bin")
    np.asarray(scale, dtype=np.float32).ravel().tofile(f"{directory}/scales.bin")
    np.asarray(cov, dtype=np.float32).ravel().tofile(f"{directory}/covs.bin")
    np.asarray(prec_sqrt, dtype=np.float32).ravel().tofile(f"{directory}/precs.bin")
    np.asarray(weights_log, dtype=np.float32).ravel().tofile(f"{directory}/weights.bin")
    np.asarray(camera_rays, dtype=np.float32).ravel().tofile(
        f"{directory}/camera_rays.bin"
    )
    np.asarray(quat_to_rot(gt_cam_quat), dtype=np.float32).ravel().tofile(
        f"{directory}/camera_rot.bin"
    )
    np.asarray(gt_cam_trans, dtype=np.float32).ravel().tofile(
        f"{directory}/camera_trans.bin"
    )
    np.asarray(view_matrix, dtype=np.float32).ravel().tofile(
        f"{directory}/camera_view_matrix.bin"
    )
    np.asarray(proj_matrix, dtype=np.float32).ravel().tofile(
        f"{directory}/camera_proj_matrix.bin"
    )
    np.asarray([tan_fovx, tan_fovy], dtype=np.float32).tofile(
        f"{directory}/tan_fovs.bin"
    )
    np.asarray([fx, fy, cx, cy], dtype=np.float32).tofile(f"{directory}/intrinsics.bin")
    np.asarray([width, height, NUM_MIXTURE], dtype=np.int32).tofile(
        f"{directory}/width_height_gaussians.bin"
    )
    # also save the transformed camera rays for reference
    camera_rays_xfm = jnp.einsum(
        "aj,jk->ak", camera_rays, gt_cam_rot_3x3
    )  # (pixels, 3)
    np.asarray(camera_rays_xfm, dtype=np.float32).ravel().tofile(
        f"{directory}/camera_rays_xfm.bin"
    )

    for var in [
        "zs",
        "alphas",
        "means",
        "quats",
        "scales",
        "covs",
        "precs",
        "weights",
        "camera_rays",
        "camera_rot",
        "camera_trans",
        "camera_view_matrix",
        "camera_proj_matrix",
        "camera_rays_xfm",
        "tan_fovs",
        "intrinsics",
    ]:
        try:
            arr = np.fromfile(f"{directory}/{var}.bin", dtype=np.float32)
        except FileNotFoundError:
            continue
        print(f"{var}[:3] in .bin: {arr.ravel()[:3]}; shape: {arr.shape}")
    print(
        f"width_height in .bin: {np.fromfile(f'{directory}/width_height_gaussians.bin', dtype=np.int32)}"
    )

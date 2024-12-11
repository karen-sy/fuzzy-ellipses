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
from scipy.spatial.transform import Rotation as R


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


root_path = "/home/karen/fuzzy-ellipses"
sys.path.append(root_path)

ycb_dir = f"{root_path}/assets/bop/ycbv/train_real"


def pose_to_matrix(pose):
    pose_rot = np.eye(4)
    pose_rot[:3, :3] = R.from_quat(pose[3:], scalar_first=True).as_matrix()
    pose_rot[:3, 3] = np.asarray(pose[:3])
    return pose_rot


def transform_points_by_inv(transform, points):
    return (
        R.from_quat(transform[3:], scalar_first=True).inv().apply(points)
        + transform[:3]
    )


class Intrinsics(NamedTuple):
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float


FRAMES = list(range(1, 1000, 300))

for scene_idx in range(1, 2):
    # scene_idx = 1
    all_data = utils.get_ycbv_data(ycb_dir, scene_idx, FRAMES, fields=[])

    # scene data
    object_idx = 1
    fx, fy, cx, cy = all_data[0]["camera_intrinsics"]
    mesh = utils.get_ycb_mesh(ycb_dir, all_data[0]["object_types"][object_idx])
    mesh_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
    _mesh_vtx = mesh.vertices  # untransformed vertices

    for frame_idx, FRAME in enumerate(FRAMES):
        # poses
        _cam_pose = all_data[frame_idx][
            "camera_pose"
        ]  # NOT identity but obj pose is in world frame already
        cam_pose = np.array(
            [
                _cam_pose[0],
                _cam_pose[1],
                _cam_pose[2],
                _cam_pose[6],
                _cam_pose[3],
                _cam_pose[4],
                _cam_pose[5],
            ]
        )  # for renderer, quat scalar should come first.
        object_pose = all_data[frame_idx]["object_poses"][
            object_idx
        ]  # pose in world frame
        object_pose_rot = pose_to_matrix(object_pose)

        # image data
        ycb_rgb = all_data[frame_idx]["rgb"]
        ycb_depth = all_data[frame_idx]["depth"]
        height, width = ycb_rgb.shape[:2]

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
        mesh_vtx_camframe = utils.transform_points(object_pose, _mesh_vtx)
        mesh_vtx_worldframe = transform_points_by_inv(cam_pose, mesh_vtx_camframe)
        print(f"{mesh_vtx_camframe.shape} vertices in mesh")
        print(f"{np.unique(mesh_vtx_camframe, axis=0).shape} unique vertices in mesh")

        x = np.array(
            fy * mesh_vtx_camframe[:, 1] / mesh_vtx_camframe[:, 2] + cy, dtype=np.int32
        )
        y = np.array(
            fx * mesh_vtx_camframe[:, 0] / mesh_vtx_camframe[:, 2] + cx, dtype=np.int32
        )
        print(f"2D x range = ({np.min(x)}, {np.max(x)})")
        print(f"2D y range = ({np.min(y)}, {np.max(y)})")
        pixels = np.stack([x, y], axis=1)
        pixels = np.clip(pixels, 0, np.array([height - 1, width - 1]))  # clip to bounds
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width, 1))

        rgb[pixels[:, 0], pixels[:, 1], :] = mesh_colors * 255
        depth[pixels[:, 0], pixels[:, 1], 0] = mesh_vtx_camframe[:, 2]

        ##### adapt setup to FMB repo boilerplate

        id_pose = np.array([0, 0, 0, 1, 0, 0, 0])
        gt_cam_trans = cam_pose[:3]
        gt_cam_quat = cam_pose[3:]

        # cam pose data
        gt_cam_4x4 = pose_to_matrix(cam_pose)

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

        # image data
        image_size = (height, width)
        color, target_depth = rgb, depth[..., 0]

        # volume usually False since color optimization implies surface samples
        # And code defaults towards that sort of usage now

        beta2 = 21.4  # how did fmb choose this?
        beta3 = 2.66

        render = fm_render.render_func_quat
        render_jit = jax.jit(fm_render.render_func_quat)

        # port FMB optimization setup
        NUM_MIXTURE = 50
        print(f"\n==========FRAME = {FRAME}==========")

        rng = np.random.default_rng(1222)
        pts = mesh_vtx_worldframe[
            rng.integers(0, len(mesh_vtx_worldframe), NUM_MIXTURE)
        ]
        gmm_init_scale = 80
        weights_log = np.log(np.ones((NUM_MIXTURE,)) / NUM_MIXTURE) + np.log(
            gmm_init_scale
        )
        mean = pts

        # Sigma = RLR^T where L diagonal elements are eigenvalues = RSS^TR^T where s = sqrt(eigenvalue)
        gaussian_quat = np.array(
            [id_pose[3:] for _ in range(NUM_MIXTURE)]
        )  # identity rotation
        _rot = jax.lax.map(
            quat_to_rot, gaussian_quat, batch_size=256
        )  # identity rotation
        scale = np.array([1 / jnp.sqrt(5e3) * np.eye(3) for _ in range(NUM_MIXTURE)])
        scale_diag = np.array([[s[0, 0], s[1, 1], s[2, 2]] for s in scale])

        rs = jnp.einsum(
            "aij,ajk->aik", _rot, scale
        )  # cov = (rot @ scale) @ (rot @ scale).T
        cov = jnp.einsum("aij,ajk->aik", rs, rs.transpose(0, 2, 1))
        prec = np.linalg.inv(cov)
        prec_sqrt = np.linalg.cholesky(
            prec
        )  # A = L L^T for real A, L lower triangular with positive diagonal

        del scale

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
            (
                np.array(np.meshgrid(np.arange(width), np.arange(height), [0]))[
                    :, :, :, 0
                ]
            )
            .reshape((3, -1))
            .T
        )
        camera_rays = (pixel_list - K[:, 2]) / np.diag(K)
        camera_rays[:, -1] = 1
        del pixel_list

        # render!
        est_depth_true = jnp.zeros((height * width))
        est_alpha_true = jnp.zeros((height * width))
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
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        _est_depth_true = np.asarray(est_depth_true)  # copy
        _est_alpha_true = np.asarray(est_alpha_true)  # copy
        axes[1].imshow(_est_alpha_true.reshape(image_size), cmap="Greys")
        axes[1].set_title("alpha @ gt pose")

        axes[2].imshow(
            jnp.where(est_alpha_true > 0.5, _est_depth_true, jnp.nan).reshape(
                image_size
            )
        )
        axes[2].set_title("depth @ gt pose (alpha > 0.5)")

        axes[1].imshow(rgb)
        axes[1].set_title("Projected GT RGB")

        axes[2].imshow(
            jnp.where(depth[..., 0] > 0, depth[..., 0], 0).reshape(image_size),
            cmap="Greys",
        )
        axes[2].set_title("Projected GT depth")

        fig.tight_layout()

        # fig.savefig(f"{directory}/python_ref.png")
        fig.savefig(f"{root_path}/_presentation_data/_python_ref_{FRAME}.png")

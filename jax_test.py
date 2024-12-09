from fuzzy_ellipses import rasterize_jit
import numpy as np
import jax.numpy as jnp
from time import time

# Test
import matplotlib.pyplot as plt

root_path = "/home/karen/fuzzy-ellipses"
image_width, image_height = 640, 480
N_MIXTURE_LIMIT = 100000

for NUM_MIXTURE in [150, 500, 2500, 5000, 10000, 50000, 100000]:
    if NUM_MIXTURE >= N_MIXTURE_LIMIT:
        break
    print(f"\n==========NUM_MIXTURE = {NUM_MIXTURE}==========")
    folder_name = f"{image_width}_{image_height}_{NUM_MIXTURE}"
    directory = f"{root_path}/data/{folder_name}"

    z_gt = np.fromfile(f"{directory}/zs.bin", dtype=np.float32).reshape(
        image_height, image_width
    )
    alphas_gt = np.fromfile(f"{directory}/alphas.bin", dtype=np.float32).reshape(
        image_height, image_width
    )
    means3D = np.fromfile(f"{directory}/means.bin", dtype=np.float32).reshape(-1, 3)
    rotations = np.fromfile(f"{directory}/quats.bin", dtype=np.float32).reshape(-1, 4)
    scales = np.fromfile(f"{directory}/scales.bin", dtype=np.float32).reshape(-1, 3)
    weights = np.fromfile(f"{directory}/weights.bin", dtype=np.float32)
    camera_rays = np.fromfile(f"{directory}/camera_rays.bin", dtype=np.float32).reshape(
        image_height, image_width, 3
    )
    camera_rot = np.fromfile(f"{directory}/camera_rot.bin", dtype=np.float32).reshape(
        3, 3
    )
    camera_trans = np.fromfile(f"{directory}/camera_trans.bin", dtype=np.float32)
    view_matrix = np.fromfile(
        f"{directory}/camera_view_matrix.bin", dtype=np.float32
    ).reshape(4, 4)
    proj_matrix = np.fromfile(
        f"{directory}/camera_proj_matrix.bin", dtype=np.float32
    ).reshape(4, 4)
    _tan_fovs = np.fromfile(f"{directory}/tan_fovs.bin", dtype=np.float32)
    tan_fovx, tan_fovy = _tan_fovs[0], _tan_fovs[1]
    _intrinsics = np.fromfile(f"{directory}/intrinsics.bin", dtype=np.float32)
    fx, fy, cx, cy, near, far = (
        _intrinsics[0],
        _intrinsics[1],
        _intrinsics[2],
        _intrinsics[3],
        _intrinsics[4],
        _intrinsics[5],
    )

    dummy = rasterize_jit(
        np.zeros_like(means3D),
        np.zeros_like(scales),
        np.zeros_like(rotations),
        np.zeros_like(weights),
        np.zeros_like(camera_rays),
        np.zeros_like(camera_rot),
        np.zeros_like(camera_trans),
        image_width,
        image_height,
        fx,
        fy,
        cx,
        cy,
        near,
        far,
    )

    start_t = time()
    z, alpha = rasterize_jit(
        means3D,
        scales,
        rotations,
        weights,
        camera_rays,
        camera_rot,
        camera_trans,
        image_width,
        image_height,
        fx,
        fy,
        cx,
        cy,
        near,
        far,
    )
    end_t = time()
    print(f"Elapsed time: {1000.0 * (end_t - start_t)} ms ({1/(end_t - start_t)} FPS)")

    viz_thresh = 0.01

    # visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _est_depth_true = np.asarray(z)  # copy
    axes[0].imshow(_est_depth_true.reshape(image_height, image_width))
    axes[0].set_title("depth @ gt pose")

    _est_alpha_true = np.asarray(alpha)  # copy
    axes[1].imshow(_est_alpha_true.reshape(image_height, image_width), cmap="Greys")
    axes[1].set_title("alpha @ gt pose")

    axes[2].imshow(
        jnp.where(_est_alpha_true > viz_thresh, _est_depth_true, jnp.nan).reshape(
            image_height, image_width
        )
    )
    axes[2].set_title(f"depth @ gt pose (alpha > {viz_thresh})")

    fig.tight_layout()

    fig.savefig(f"{directory}/jax_out.png")

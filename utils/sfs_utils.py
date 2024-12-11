import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.spatial.transform import Rotation as R
from typing import NamedTuple


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
    vmin=None,
    vmax=None,
    cmap=None,
    interp="nearest",
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(
        rows, cols, gridspec_kw=gridspec_kw, figsize=(cols * 4, rows * 4)
    )
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3], interpolation=interp)
        else:
            # only render Alpha channel
            ax.imshow(im[...], vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interp)
        if not show_axes:
            ax.set_axis_off()
    plt.tight_layout()


class DegradeLR:
    def __init__(
        self,
        init_lr,
        p_thresh=5e-2,
        window=10,
        p_window=5,
        slope_less=0,
        max_drops=4,
        print_debug=True,
    ):
        assert (init_lr > 0) and (p_thresh > 0) and (p_thresh < 1)
        self.init_lr = init_lr
        self.p_thresh = p_thresh
        self.window = int(round(window))
        if self.window < 3:
            print("window too small! clipped to 3")
            self.window = 3
        self.slope_less = slope_less
        self.p_window = int(round(p_window))
        if self.p_window < 1:
            print("p_window too small! clipped to 1")
            self.p_window = 1
        self.train_val = []
        self.prior_p = []
        self.n_drops = 0
        self.max_drops = max_drops
        self.last_drop_len = self.window + 1
        self.step_func = lambda x: self.init_lr / (10**self.n_drops)
        self.print_debug = print_debug
        self.counter = 0

    def add(self, error):
        self.counter += 1
        self.train_val.append(error)
        len_of_opt = len(self.train_val)

        if len_of_opt >= self.window + self.p_window:
            yo = np.array(self.train_val[-self.window :])
            yo = yo / yo.mean()
            xo = np.arange(self.window)
            xv = np.vstack([xo, np.ones_like(xo)]).T
            w = np.linalg.pinv(xv.T @ xv) @ xv.T @ yo
            yh = xo * w[0] + w[1]
            var = ((yh - yo) ** 2).sum() / (self.window - 2)
            var_slope = (12 * var) / (self.window**3)
            ps = 0.5 * (1 + erf((self.slope_less - w[0]) / (np.sqrt(2 * var_slope))))
            self.prior_p.append(ps)

            p_eval = np.array(self.prior_p[-self.p_window :])
            if (p_eval < self.p_thresh).all():
                self.n_drops += 1
                if self.n_drops > self.max_drops:
                    if self.print_debug:
                        print("early exit due to max drops")
                    return True
                if self.print_debug:
                    print(
                        "dropping LR to {:.2e} after {} steps".format(
                            self.step_func(0), self.counter - 1
                        )
                    )
                min_len = self.window + self.p_window
                if self.last_drop_len == min_len and len_of_opt == min_len:
                    if self.print_debug:
                        print("early exit due to no progress")
                    return True
                self.last_drop_len = len(self.train_val)
                self.train_val = []
        return False


## Helpers.
# converts quaternion to rotation matrix
def quat_to_rot(q):
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z

    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    R1 = jnp.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )
    R2 = jnp.eye(3)
    return jnp.where(Nq > 1e-12, R1, R2)


def pose_to_matrix(pose):
    pose_rot = np.eye(4)
    pose_rot[:3, :3] = R.from_quat(pose[3:], scalar_first=True).as_matrix()
    pose_rot[:3, 3] = np.asarray(pose[:3])
    return pose_rot


def transform_points_jittable(transform, points):
    return jnp.einsum("ij,kj->ki", quat_to_rot(transform[3:]), points) + transform[:3]


def transform_points_by_inv(transform, points):
    return (
        R.from_quat(transform[3:], scalar_first=True).inv().apply(points)
        + transform[:3]
    )


def transform_points_by_inv_jittable(transform, points):
    ret = jnp.einsum("ij,kj->ki", quat_to_rot(transform[3:]).T, points) + transform[:3]
    return ret


def transform_rays(camera_rays, quat, t):
    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None], (camera_rays.shape[0], 1))

    camera_starts_rays = jnp.stack([camera_rays, trans], 1)
    return camera_starts_rays


class Intrinsics(NamedTuple):
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float


def prec_from_rotscale(rotations, scales):
    _rot_mtx = jnp.array([quat_to_rot(q) for q in rotations])
    _scale_mtx = jnp.array([jnp.diag(s) for s in scales])
    _rs = jnp.einsum("aij,ajk->aik", _rot_mtx, _scale_mtx)
    _cov = jnp.einsum("aij,ajk->aik", _rs, _rs.transpose(0, 2, 1))
    prec = jnp.linalg.cholesky(jnp.linalg.inv(_cov))
    return prec

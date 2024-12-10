import torch
from fuzzy_ellipses import _C

import functools
import jax
from jax import core, dtypes
from jax.core import ShapedArray
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir, xla
from jax.lib import xla_client
import numpy as np
import jax.numpy as jnp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NUM_CHANNELS = 5

################################
# Helpers and boilerplates
################################

for _name, _value in _C.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


# XLA array layout in memory
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


################################
# Data helpers
################################


def getProjectionMatrixJax(width, height, fx, fy, cx, cy, znear, zfar):
    fovX = jnp.arctan(width / 2 / fx) * 2.0
    fovY = jnp.arctan(height / 2 / fy) * 2.0

    tanHalfFovY = jnp.tan((fovY / 2))
    tanHalfFovX = jnp.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0
    P = jnp.transpose(
        jnp.array(
            [
                [
                    2.0 * znear / (right - left),
                    0.0,
                    (right + left) / (right - left),
                    0.0,
                ],
                [
                    0.0,
                    2.0 * znear / (top - bottom),
                    (top + bottom) / (top - bottom),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    z_sign * zfar / (zfar - znear),
                    -(zfar * znear) / (zfar - znear),
                ],
                [0.0, 0.0, z_sign, 0.0],
            ]
        )
    )
    return P


################################
# Rasterize fwd primitive
################################


def _build_rasterize_gaussians_fwd_primitive():
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(
        means3D,
        weights,
        scales,
        rotations,
        camera_rays,
        camera_rot,
        camera_trans,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
    ):
        float_dtype = dtypes.canonicalize_dtype(np.float32)

        return [
            ShapedArray((image_height, image_width), float_dtype),
            ShapedArray((image_height, image_width), float_dtype),
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(
        ctx,
        means3D,
        weights,
        scales,
        rotations,
        camera_rays,
        camera_rot,
        camera_trans,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
    ):
        float_to_ir = mlir.dtype_to_ir_type(np.dtype(np.float32))

        num_gaussians = ctx.avals_in[1].shape[0]
        opaque = _C.build_gaussian_rasterize_descriptor(
            image_height, image_width, num_gaussians, tanfovx, tanfovy
        )

        op_name = "rasterize_gaussians_fwd"

        operands = [
            means3D,
            weights,
            scales,
            rotations,
            camera_rays,
            camera_rot,
            camera_trans,
            viewmatrix,
            projmatrix,
        ]

        operands_ctx = ctx.avals_in[: len(operands)]

        output_shapes = [(image_height, image_width), (image_height, image_width)]

        result_types = [
            mlir.ir.RankedTensorType.get([image_height, image_width], float_to_ir),
            mlir.ir.RankedTensorType.get([image_height, image_width], float_to_ir),
        ]

        return custom_call(
            op_name,
            # Output types
            result_types=result_types,
            # The inputs:
            operands=operands,
            backend_config=opaque,
            operand_layouts=default_layouts(*[i.shape for i in operands_ctx]),
            result_layouts=default_layouts(*output_shapes),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive("jax_render_primitive_fwd")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_fwd_lowering, platform="gpu")
    _rasterize_prim.def_abstract_eval(_rasterize_fwd_abstract)

    return _rasterize_prim


rasterizer_fwd_primitive = _build_rasterize_gaussians_fwd_primitive()

################################
# Main API
################################


def rasterize(
    means3D,
    scales,
    rotations,
    weights,
    camera_rays,
    camera_rot,
    camera_trans,
    image_height,
    image_width,
    fx,
    fy,
    cx,
    cy,
    near,
    far,
):
    fovX = np.arctan(image_width / 2 / fx) * 2.0
    fovY = np.arctan(image_height / 2 / fy) * 2.0
    tan_fovx = np.tan(fovX)
    tan_fovy = np.tan(fovY)

    pmatrix = getProjectionMatrixJax(
        image_width, image_height, fx, fy, cx, cy, near, far
    )
    view_matrix = jnp.transpose(jnp.linalg.inv(jnp.eye(4)))

    proj_matrix = view_matrix @ pmatrix

    z, alpha = rasterizer_fwd_primitive.bind(
        means3D,
        weights,
        scales,
        rotations,
        camera_rays,
        camera_rot,
        camera_trans,
        view_matrix,
        proj_matrix,
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        image_height=image_height,
        image_width=image_width,
    )
    z = jnp.nan_to_num(z, posinf=10.0, neginf=0.0)
    alpha = jnp.nan_to_num(alpha, posinf=1, neginf=0)

    return (z, alpha)


rasterize_jit = jax.jit(rasterize, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14))

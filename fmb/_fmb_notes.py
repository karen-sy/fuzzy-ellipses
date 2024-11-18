import jax
import jax.numpy as jnp
from baseline.util_render import jax_stable_exp


# core rendering function
def render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3):
    # precision is fully parameterized by triangle matrix
    # we use upper triangle for compatibilize with sklearn
    prec = jnp.triu(prec_full)

    # if doing normalized weights for proper GMM
    # typically not used for shape reconstruction
    # weights = jnp.exp(weights_log)
    # weights = weights/weights.sum()

    """
        perf_idx, perf_ray should be kernel code
    """

    # gets run per gaussian with [precision, log(weight), mean]
    def perf_idx(prcI, w, meansI):
        # math is easier with lower triangle
        prc = prcI.T  # (3, 3)

        # gaussian scale
        # could be useful for log likelihood but not used here
        # div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20

        # gets run per ray
        def perf_ray(r_t):
            # unpack the ray (r) and position (t)
            r = r_t[0]  # (3, )
            t = r_t[1]  # (3, )

            # shift the mean to be relative to ray start
            p = meansI - t  # (3, )
            print(f"p={p.shape}")

            # compute \sigma^{-0.5} p, which is reused
            projp = prc @ p  # (3,3)@(3,) = (3, )
            print(f"projp={projp}")

            # compute v^T \sigma^{-1} v
            vsv = ((prc @ r) ** 2).sum()  # (prc @ r) is (3,3)@(3,) = (3, )

            # compute p^T \sigma^{-1} v
            psv = ((projp) * (prc @ r)).sum()  # (proj * (prc @ r)) is (3,)
            # jax.debug.print("psv={psv}, projp={projp}, prc={prc}, r={r}", psv=psv, projp=projp, prc=prc, r=r)

            # # compute the surface normal as \sigma^{-1} p
            # projp2 = prc.T @ projp  # (3, )

            # distance to get maximum likelihood point for this gaussian
            # scale here is based on r!
            # if r = [x, y, 1], then depth. if ||r|| = 1, then distance
            z = (psv) / (vsv)  # float

            # get the intersection point
            v = r * z - p  # (3, )

            # compute intersection's unnormalized Gaussian log likelihood
            d0 = ((prc @ v) ** 2).sum()  # + 3*jnp.log(jnp.pi*2)   # float

            # multiply by the weight
            std = -0.5 * d0 + w  # float

            # if you wanted real probability
            # d3 =  std + jnp.log(div) #+ 3*jnp.log(res)

            # # compute a normalized normal
            # norm_est = projp2 / jnp.linalg.norm(projp2)  # (3, )
            # norm_est = jnp.where(r @ norm_est < 0, norm_est, -norm_est)

            # return ray distance, gaussian distance, normal
            return z, std

        # runs parallel for each ray across each gaussian
        z, std = jax.vmap((perf_ray))(camera_starts_rays)

        return z, std  # (num_pixels,), (num_pixels,),

    # runs parallel for gaussian
    zs, stds = jax.vmap(perf_idx)(
        prec, weights_log, means
    )  # (num_gaussians, num_pixels,), (num_gaussians, num_pixels,), (num_gaussians, num_pixels, 3)

    """
        the below are all elementwise processing on (N_G, N_P) or (N_P,) arrays.
        Either push them into the kernel, or use thrust to do simple elementwise processing/
    """
    # alpha is based on distance from all gaussians
    est_alpha = 1 - jnp.exp(-jnp.exp(stds).sum(0))  # (num_pixels,)  # Eq.8

    # points behind camera should be zero
    # BUG: est_alpha should also use this
    sig1 = zs > 0  # sigmoid

    # compute the algrebraic weights in the paper  (Eq. 7)
    w_intersection = (
        sig1 * jnp.nan_to_num(jax_stable_exp(-zs * beta_2 + beta_3 * stds)) + 1e-20
    )

    # normalize weights
    wgt = w_intersection.sum(0)  # (num_pixels,)  # ATOMICADD into this (default 0)
    # w_intersection = w_intersection / jnp.where(wgt == 0, 1, wgt)  # after kernel

    # compute weighted z and normal
    zs_final = (w_intersection * jnp.nan_to_num(zs)).sum(0) / jnp.where(
        wgt == 0, 1, wgt
    )  # ATOMICADD into this (default 0)

    # return z, alpha (skip normal for minimal repro)
    # weights can be used to compute color, DINO features, or any other per-Gaussian property
    return zs_final, est_alpha  # (num_pixels,), (num_pixels,)

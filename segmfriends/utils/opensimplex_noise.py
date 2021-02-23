import numpy as np
try:
    from nifty.external import generate_opensimplex_noise
except ImportError:
    generate_opensimplex_noise = None

def add_opensimplex_noise_to_affs(affinities, scale_factor,
                            mod='merge-biased',
                            target_affs='all',
                            seed=None,
                            number_of_threads=-1
                            ):
    assert generate_opensimplex_noise is not None, "Opensimplex noise is needed for this function"
    affinities = affinities.copy()

    if target_affs == 'short':
        # TODO: fix this by detecting actual direct neighbors
        noise_slc = slice(0, 3)
    elif target_affs == 'long':
        noise_slc = slice(3, None)
    elif target_affs == "all":
        noise_slc = slice(None)
    else:
        raise ValueError


    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def logit(x, clip=True):
        if clip:
            x = add_epsilon(x)
        return np.log(x / (1. - x))

    def add_epsilon(affs, eps=1e-2):
        p_min = eps
        p_max = 1. - eps
        return (p_max - p_min) * affs + p_min

    # Generate noise:
    shape = affinities[noise_slc].shape

    large_ft_size = np.array((17., 3., 50., 50.))
    large_scale_noise = (generate_opensimplex_noise(shape, seed=seed, features_size=large_ft_size, number_of_threads=number_of_threads)
                         +1.0) / 2.0
    fine_ft_size = np.array((17., 3., 20., 20.))
    fine_scale_noise = (generate_opensimplex_noise(shape, seed=seed, features_size=fine_ft_size, number_of_threads=number_of_threads)
                        + 1.0) / 2.0

    # Combine large and fine features:
    # TODO: more or simplify?
    large_scale, fine_scale = 10, 5
    simplex_noise = (large_scale_noise * large_scale + fine_scale_noise * fine_scale) / (large_scale + fine_scale)

    if mod == "merge-biased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(np.maximum(simplex_noise, 0.5)))
    elif mod == "split-biased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(np.minimum(simplex_noise, 0.5)))
    elif mod == "unbiased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(simplex_noise))
    else:
        raise ValueError("Accepted mods are add or subtract")

    affinities[noise_slc] = noisy_affs

    return affinities

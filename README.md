# Segmfriends
Bunch of tools and experiments for image segmentation


## Install
### Basic version
This version can run most of the clustering/segmentation algorithms included:

- `conda create -n segmFr -c abailoni -c conda-forge nifty vigra cython`
- `source activate segmFr`
- `python setup.py install` (or `python setup.py develop` if you plan to work on the package)

### Full version
For running all deep-learning experiments, at the moment the following dependencies are required:
- `conda create -n segmFr -c abailoni -c conda-forge -c pytorch nifty vigra cython inferno`
    - See `inferno` package here: https://github.com/inferno-pytorch/inferno
- `firelight`: https://github.com/inferno-pytorch/firelight
- `speedrun`: https://github.com/inferno-pytorch/speedrun/tree/new_import (branch `new_import`)
- `neurofire`: https://github.com/abailoni/neurofire (my fork soon merged)
- `ConfNets`: https://github.com/imagirom/ConfNets/tree/multi-scale-unet - (branch `multi-scale-unet`)
- `affogato`: https://github.com/constantinpape/affogato/tree/affinities-with-glia (branch `affinities-with-glia`)
- (for evaluation scores, install module `cremi`: https://github.com/constantinpape/cremi_python/tree/py3)

Coming soon: `segmfriend` conda-package

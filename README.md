# Segmfriends
Bunch of tools and experiments for image segmentation (still in the research-code stage)


## Install
The basic version of the package can be installed with:

- `conda create -n segmFr -c abailoni -c conda-forge nifty vigra cython`
- `source activate segmFr`
- `python setup.py install` (or `python setup.py develop` if you plan to work on the package)


However, for running all experiments, further requirements need to be installed:
- `inferno` (and `torch`): https://github.com/inferno-pytorch/inferno
- `neurofire`: https://github.com/inferno-pytorch/neurofire
- `speedrun`: https://github.com/inferno-pytorch/speedrun
- `firelight`
- `ConfNets`
- `affogato`
- (`cremi`: https://github.com/constantinpape/cremi_python/tree/py3)

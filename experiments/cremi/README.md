# Experiments on CREMI dataset
To run a `speedrun` experiment and train affinities, run the following command from the main folder of the package:

`CUDA_VISIBLE_DEVICES=1 python experiments/cremi/train_affinities.py -- name_experiment --inherit example_train_affs.yml`

The experiment data are saved by default in the `experiments/cremi/runs` folder.

To start a new training and loading a previous model, run for example the following command:

`CUDA_VISIBLE_DEVICES=1 python experiments/cremi/train_affinities.py -- name_new_experiment --inherit example_train_affs.yml
  --update0 new_experiment_config.yml --config.model.model_kwargs.loadfrom PATH_TO_OLD_CHECKPOINT.pytorch --config.trainer.optimizer.Adam.lr 6e-5`

For more details about the CREMI dataset and applied augmentations, see docstring of `segmfriends.datasets.cremi.CremiDataset`




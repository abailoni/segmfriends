### Comparing post-processing pipelines on affinities predicted by a CNN
The `compare_postprocessing_from_affs.py` script is used to run post-processing pipelines on **affinities** predicted by a CNN (general graphs not supported yet).

The script lets you run several multiple post-processing pipelines at the same time in a single run (multi-processing is supported). Everything is setup via a config file (thanks to [speedrun](https://github.com/inferno-pytorch/speedrun)).

Examples of post-processing pipelines are:
- GASP (with support for superpixel generation)
- Mutex Watershed
- Multicut (with support for superpixel generation)

Please refer to the file `postproc_configs/main_config.yml` for a detailed description of all the options available.

### Comparing post-processing pipelines on several affinities predicted by different CNN models
Often you will want to run post-processing on multiple types of affinities predicted by different CNN models. This is also fully supported, as explained in the comments of the `postproc_configs/main_config.yml`. You may want to use the script `run_postprocessing_on_multiple_affs.py` to simplify and automatize the process. 

### Understanding the mess
First of all, this script is used for running post_proc on **affinities** (no general graphs so far).


- `main_output_dir` defines the main directory where all the sub-experiments you will run will be saved (including data like scores and segmentations)
-   
- In general, you could have different versions of affinities (predicted by several models, etc..). All these affinities should be placed in subfolders of `proj_dir_path`

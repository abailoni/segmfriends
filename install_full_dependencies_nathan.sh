# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# TODO: uncomment, create req.yaml?
# Only install the dependencies (we will install inferno from the latest github repo):
conda activate <YOUR_ENV_NAME>
conda install --only-deps -c pytorch -c abailoni -c conda-forge cython firelight inferno
conda install -c abailoni -c conda-forge cython firelight

# Install extra repositories in sub-folder or download these somewhere else:
if [ ! -d "./downloads" ]; then mkdir ./downloads; fi
WORKING_DIR=./downloads/deps
if [ -d "$WORKING_DIR" ]; then rm -Rf $WORKING_DIR; fi
mkdir $WORKING_DIR
cd $WORKING_DIR

git clone https://github.com/abailoni/inferno.git
cd ./inferno
git checkout my_dev
python setup.py develop
cd ..


git clone https://github.com/abailoni/speedrun.git
cd ./speedrun
git checkout inference-engine
python setup.py develop
cd ..

git clone https://github.com/abailoni/neurofire.git
cd ./neurofire
python setup.py develop
cd ..

git clone https://github.com/imagirom/ConfNets.git
cd ConfNets
git checkout multi-scale-unet
python setup.py develop
cd ..

git clone https://github.com/constantinpape/cremi_python.git
cd cremi_python
git checkout py3
python setup.py install
cd ..

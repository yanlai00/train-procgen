## Setup 
Need to export variables:
first confirm `/usr/local/cuda-10.0/lib64` exists (or other versions of cuda, e.g. cuda-5.0), then
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib
```
if the desired GPU has index 0, do: 
```
export CUDA_VISIBLE_DEVICES=0
```

As a reference, output from current conda env that uses GPU:
```
(train-procgen)$ echo $LD_LIBRARY_PATH
/home/mandi/.local/bin:/home/mandi/bin:/home/mandi/miniconda3/envs/train-procgen/bin:/home/mandi/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/ussr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/mandi/bin:/home/mandi/bin:/home/mandi/lib:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib
(train-procgen) $ echo $CUDA_VISIBLE_DEVICES 
0
```

## experiments to run
1. train (and save) an agent on randomly random cut frames 
```
conda activate train-procgen
mkdir log
python train_procgen/train_crop.py -id 0 --use "all" --num_levels 50
## If you want to use taskset to limit CPU usage, say only use CPU index 0-5, do
## taskset -c 0-5 python train_procgen/train_recenter.py --nupdates 0 -id 0
## same for commands below
```


**Status:** Archive (code is provided as-is, no updates expected)

## Leveraging Procedural Generation to Benchmark Reinforcement Learning

#### [[Blog Post]](https://openai.com/blog/procgen-benchmark/) [[Paper]](https://arxiv.org/abs/1912.01588)

This is code for training agents for some of the experiments in [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://cdn.openai.com/procgen.pdf) [(citation)](#citation).  The code for the environments is in the [Procgen Benchmark](https://github.com/openai/procgen) repo.

Supported platforms:

- macOS 10.14 (Mojave)
- Ubuntu 16.04

Supported Pythons:

- 3.7 64-bit

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from [`environment.yml`](environment.yml) manually.

```
git clone https://github.com/openai/train-procgen.git
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
pip install -e train-procgen
```

# Citation

Please cite using the following bibtex entry:

```
@article{cobbe2019procgen,
  title={Leveraging Procedural Generation to Benchmark Reinforcement Learning},
  author={Cobbe, Karl and Hesse, Christopher and Hilton, Jacob and Schulman, John},
  journal={arXiv preprint arXiv:1912.01588},
  year={2019}
}
```

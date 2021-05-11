
# Leveraging Procedural Generation to Benchmark Reinforcement Learning

## Setup
```
git clone https://github.com/yanlai00/train-procgen.git
cd train-procgen
conda env create -f environment.yml
pip install -e .
```

## Example command-line usage 
1. To run a training experiments with 50 avaliable levels for 20M steps and log every 10 steps, use random crop augmentation on top of the baseline PPO algorithm, and saved the model with idx 1:
```
python train_procgen/train.py -id 1 --use cross --dropout 0.5 --netrand 1

```

2. To test a trained random-random-cut agent at index 3 on a set of {100, 1000, 2000, ..., 95000} level intervals:
(it's easier to put these blocks of commands in a bash .sh file and run from command line)

```
LOAD=2
USE="recenter"
python train_procgen/test.py --start_level 1000 -id 0 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 10000 -id 1 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 20000 -id 2 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 30000 -id 3 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 40000 -id 4 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 50000 -id 5 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 60000 -id 6 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 70000 -id 7 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 80000 -id 8 --load_id ${LOAD} --use ${USE}
python train_procgen/test.py --start_level 90000 -id 9 --load_id ${LOAD} --use ${USE}
```

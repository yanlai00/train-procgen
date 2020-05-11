Colab Setup:

```
!pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
!pip install tensorflow==1.15.0 mpi4py==3.0.3 gym==0.15.4 
!git clone https://github.com/openai/procgen.git
!git clone https://github.com/MandiZhao/train-procgen.git
%cd procgen
!pip install -e .
%cd ..
%cd train-procgen
!pip install -e .
```



Before anything:
```
conda activate train-procgen
```

To train:

```
taskset -c 0-6 train_procgen/train.py 
taskset -c 0-6 train_procgen/train_random.py 
```

To test with a saved model:
```
taskset -c 0-6 train_procgen/test_random.py -nroll 20
```


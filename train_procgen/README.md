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


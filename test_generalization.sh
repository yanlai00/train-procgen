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
#!/bin/bash

python train_base.py --transfer_task "[[1], [0]]"
python train_base.py --transfer_task "[[1], [2]]"
python train_base.py --transfer_task "[[2], [0]]"
python train_base.py --transfer_task "[[2], [1]]"



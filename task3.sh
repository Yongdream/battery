#!/bin/bash

#tasks=(
#  "[[0], [0]]"
#  "[[2], [2]]"
#  "[[1], [1]]"
#)
tasks=(
  "[[0], [1]]"
  "[[0], [2]]"
#
  "[[1], [0]]"
  "[[1], [2]]"
#
  "[[2], [0]]"
  "[[2], [1]]"

  "[[0], [0]]"
  "[[1], [1]]"
  "[[2], [2]]"
)

num_iterations=3
seed=13

for ((i=0; i<$num_iterations; i++)); do
  let "seed=seed+5"
  for task in "${tasks[@]}"; do

    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_loss "DSAN" --seed $seed

  done
done

# ref: [Windows系统在pycharm上运行.sh文件]https://zhuanlan.zhihu.com/p/609906237



#!/bin/bash

tasks=(
  "[[0], [1]]"
  "[[0], [2]]"
  "[[1], [0]]"
  "[[1], [2]]"
  "[[2], [0]]"
  "[[2], [1]]"
)
#tasks=(
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#  "[[2], [0]]"
#)

for task in "${tasks[@]}"; do
  python train_base.py --transfer_task "$task" --distance_loss 'MK-MMD'
  python train_base.py --transfer_task "$task" --distance_loss 'CORAL'
done

# ref: [Windows系统在pycharm上运行.sh文件]https://zhuanlan.zhihu.com/p/609906237



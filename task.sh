#!/bin/bash


tasks=(

  "[[2], [1]]"
  "[[2], [0]]"

#  "[[0], [1]]"
#  "[[0], [2]]"
#
#  "[[1], [0]]"
#  "[[1], [2]]"

#  "[[0], [0]]"
#  "[[1], [1]]"
#  "[[2], [2]]"

)

num_iterations=5
seed=10

for ((i=0; i<$num_iterations; i++)); do
  let "seed=seed+5"
  for task in "${tasks[@]}"; do

    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_loss "DSAN" --lam_distance 3 --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeaturesNoAtt" --distance_loss "DSAN" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeaturesNoSign" --distance_loss "DSAN" --seed $seed
#
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_loss "JMMD" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_loss "MK-MMD" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_loss "CORAL" --seed $seed
#
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_metric "False" --domain_adversarial "True" --adversarial_loss "DA" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_metric "False" --domain_adversarial "True" --adversarial_loss "CDA" --seed $seed
#
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeatures" --distance_metric "False" --domain_adversarial "False" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeaturesNoSign" --distance_loss "DSAN" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "ALSTMAdFeaturesNoAtt" --distance_loss "DSAN" --seed $seed
#
#    python train_base.py --transfer_task "$task" --model_name "WDCNNModel" --distance_metric "False" --seed $seed
#    python train_base.py --transfer_task "$task" --model_name "CBDANModel" --domain_adversarial 'True' --distance_metric 'False' --adversarial_loss "DA" --seed $seed

  done
done

# ref: [Windows系统在pycharm上运行.sh文件]https://zhuanlan.zhihu.com/p/609906237



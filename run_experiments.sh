#!/usr/bin/env bash
# Experiment suite using ONLY syllabus architectures.
# Every run = 5-fold subject-independent CV; results -> results/results.jsonl
#   bash run_experiments.sh stage1          # A, B, D
#   bash run_experiments.sh stage2 <enc>    # C: pre-training + 4-modality fusion with best encoder
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python
T="$PY src/train.py"
lr_of() { case $1 in transformer|vit|bert) echo 1e-3;; *) echo 2e-3;; esac; }

if [ "${1:-stage1}" = stage1 ]; then
  echo "### A. Single-modality baselines (ResNet)"
  for m in IMU FP EMG EEG; do $T --mods $m --encoder resnet --name A_${m}_resnet; done

  echo "### B. Architecture comparison on IMU+FP+EMG fusion"
  for e in mlp mlp_wn vgg inception resnet resnet_dilated densenet mobilenet efficientnet convnext \
           rnn gru lstm lstm_uni transformer vit bert; do
    $T --mods IMU FP EMG --encoder $e --lr $(lr_of $e) --name B_$e
  done

  echo "### D. Optimizer study (IMU, ResNet, 25 epochs)"
  for spec in sgd:0.05 momentum:0.02 nesterov:0.02 adagrad:0.02 rmsprop:1e-3 adam:2e-3 adamw:2e-3; do
    o=${spec%%:*}; lr=${spec##*:}
    $T --mods IMU --encoder resnet --opt $o --lr $lr --epochs 25 --patience 25 --name D_opt_$o
  done
  echo STAGE1_DONE
else
  E=$2
  echo "### C. Generative pre-training (${E}) + 4-modality fusion"
  for p in ae dae sae vae; do
    $T --mods IMU FP EMG --encoder $E --lr $(lr_of $E) --pretrain $p --name C_${E}_$p
  done
  $T --mods IMU FP EMG EEG --encoder $E --lr $(lr_of $E) --name C_${E}_4mod

  echo "### G. Conditional-GAN data augmentation"
  $T --mods IMU FP EMG --encoder $E --lr $(lr_of $E) --gan_aug 0.5 --name G_${E}_ganaug
  $T --mods IMU --encoder resnet --gan_aug 0.5 --name G_IMU_resnet_ganaug     # vs A_IMU_resnet
  $T --mods IMU FP EMG --encoder $E --lr $(lr_of $E) --gan_only --name G_${E}_TSTR
  echo STAGE2_DONE
fi

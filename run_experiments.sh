#!/usr/bin/env bash
# Experiment suite using ONLY syllabus architectures.
# Every run = 5-fold subject-independent CV; results -> results/results.jsonl
#   bash run_experiments.sh                 # run stage1 + stage2 (incl. anomaly & forecast) + report
#   bash run_experiments.sh stage1          # A, B, D
#   bash run_experiments.sh stage2 [enc]    # C, E, F, G + report
#   bash run_experiments.sh anomaly [enc]   # E: Sparse AE, Conv-AE, BiGAN (MSE/MAE, AUC-ROC, F1)
#   bash run_experiments.sh forecast        # F: AE+LSTM, cGAN, Multi-task Transformer (RMSE, MAE, Ramp, Peak, CRPS)
#   bash run_experiments.sh report          # build summary.md, ensemble metrics, and plots
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python ]; then
  PY=.venv/Scripts/python
else
  PY=${PYTHON:-python3}
fi

ensure_data() {
  if [ ! -d data/processed/IMU ] || [ -z "$(ls -A data/processed/IMU 2>/dev/null || true)" ]; then
    echo "### Preprocessing raw dataset into data/processed/ ..."
    $PY src/prepare_data.py
  fi
}

T="$PY src/train.py"
lr_of() { case $1 in transformer|vit|bert|mt_transformer) echo 1e-3;; *) echo 2e-3;; esac; }

run_stage1() {
  echo "### A. Single-modality baselines (ResNet)"
  for m in IMU FP EMG EEG; do $T --mods $m --encoder resnet --name A_${m}_resnet; done

  echo "### B. Architecture comparison on IMU+FP+EMG fusion"
  for e in mlp mlp_wn vgg inception resnet resnet_dilated densenet mobilenet efficientnet convnext \
           rnn gru lstm lstm_uni transformer vit bert conv_ae ae_lstm mt_transformer; do
    $T --mods IMU FP EMG --encoder $e --lr $(lr_of $e) --name B_$e
  done

  echo "### D. Optimizer study (IMU, ResNet, 25 epochs)"
  for spec in sgd:0.05 momentum:0.02 nesterov:0.02 adagrad:0.02 rmsprop:1e-3 adam:2e-3 adamw:2e-3; do
    o=${spec%%:*}; lr=${spec##*:}
    $T --mods IMU --encoder resnet --opt $o --lr $lr --epochs 25 --patience 25 --name D_opt_$o
  done
  echo STAGE1_DONE
}

run_anomaly() {
  E=${1:-$([ -f results/results.jsonl ] && $PY src/pick.py best || echo resnet)}
  echo "### E. Unsupervised Reconstruction & Anomaly Detection (Sparse AE, Conv-AE, BiGAN)"
  $T --task anomaly --mods IMU FP EMG --encoder "$E" --lr $(lr_of "$E") --pretrain sae --name E_sparse_ae
  $T --task anomaly --mods IMU FP EMG --encoder conv_ae --lr 2e-3 --pretrain conv_ae --name E_conv_ae
  $T --task anomaly --mods IMU FP EMG --encoder "$E" --lr $(lr_of "$E") --pretrain bigan --name E_bigan
}

run_forecast() {
  echo "### F. Continuous Load Forecasting & Multi-Task Probabilistic Models (AE+LSTM, cGAN, Multi-task Transformer)"
  $T --task forecast --mods IMU FP EMG --forecaster ae_lstm --lr 2e-3 --name F_ae_lstm
  $T --task forecast --mods IMU FP EMG --forecaster cgan --lr 1e-3 --name F_cgan
  $T --task forecast --mods IMU FP EMG --forecaster mt_transformer --lr 1e-3 --name F_mt_transformer
}

run_stage2() {
  E=${1:-$([ -f results/results.jsonl ] && $PY src/pick.py best || echo resnet)}
  echo "### C. Generative pre-training (${E}) + 4-modality fusion"
  for p in ae dae sae conv_ae vae bigan; do
    $T --mods IMU FP EMG --encoder "$E" --lr $(lr_of "$E") --pretrain $p --name C_${E}_$p
  done
  $T --mods IMU FP EMG EEG --encoder "$E" --lr $(lr_of "$E") --name C_${E}_4mod

  run_anomaly "$E"
  run_forecast

  echo "### G. Conditional-GAN data augmentation"
  $T --mods IMU FP EMG --encoder "$E" --lr $(lr_of "$E") --gan_aug 0.5 --name G_${E}_ganaug
  $T --mods IMU --encoder resnet --gan_aug 0.5 --name G_IMU_resnet_ganaug     # vs A_IMU_resnet
  $T --mods IMU FP EMG --encoder "$E" --lr $(lr_of "$E") --gan_only --name G_${E}_TSTR
  echo STAGE2_DONE
}

run_report() {
  ENS=$([ -f results/results.jsonl ] && $PY src/pick.py ensemble || echo "")
  $PY src/report.py $ENS
}

MODE=${1:-all}
case "$MODE" in
  stage1)
    ensure_data
    run_stage1
    ;;
  stage2)
    ensure_data
    run_stage2 "${2:-}"
    run_report
    ;;
  anomaly)
    ensure_data
    run_anomaly "${2:-}"
    run_report
    ;;
  forecast)
    ensure_data
    run_forecast
    run_report
    ;;
  report)
    run_report
    ;;
  all)
    ensure_data
    run_stage1
    run_stage2 "${2:-}"
    run_report
    ;;
  *)
    echo "Usage: bash run_experiments.sh [all|stage1|stage2 [enc]|anomaly [enc]|forecast|report]"
    exit 1
    ;;
esac

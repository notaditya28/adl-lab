## All runs (5-fold subject-independent CV)

| Run | Modalities | Encoder | Optimizer | Params | Window acc | Window F1 | Trial acc |
|---|---|---|---|---|---|---|---|
| C_fusion3_cnn_dae | IMU+FP+EMG | cnn | adamw | 0.79M | 97.40 ± 3.40 | 97.40 | 97.48 |
| B_fusion3_transformer | IMU+FP+EMG | transformer | adamw | 2.97M | 96.74 ± 0.66 | 96.73 | 97.48 |
| B_fusion3_cnn | IMU+FP+EMG | cnn | adamw | 0.79M | 96.39 ± 1.94 | 96.33 | 96.23 |
| B_fusion3_lstm | IMU+FP+EMG | lstm | adamw | 1.70M | 95.90 ± 0.94 | 95.94 | 96.86 |
| C_fusion4_cnn | IMU+FP+EMG+EEG | cnn | adamw | 0.93M | 95.20 ± 4.44 | 95.20 | 95.57 |
| A_FP_cnn | FP | cnn | adamw | 0.16M | 93.29 ± 3.19 | 93.25 | 93.64 |
| B_fusion3_gru | IMU+FP+EMG | gru | adamw | 1.42M | 91.63 ± 4.85 | 91.62 | 91.82 |
| D_opt_adam | IMU | cnn | adam | 0.24M | 91.31 ± 2.94 | 91.29 | 92.81 |
| A_IMU_cnn | IMU | cnn | adamw | 0.24M | 91.19 ± 4.57 | 91.19 | 92.81 |
| D_opt_nesterov | IMU | cnn | nesterov | 0.24M | 89.82 ± 3.99 | 89.79 | 91.62 |
| D_opt_momentum | IMU | cnn | momentum | 0.24M | 89.68 ± 4.06 | 89.65 | 91.02 |
| D_opt_adamw | IMU | cnn | adamw | 0.24M | 88.97 ± 4.93 | 88.99 | 89.82 |
| D_opt_sgd | IMU | cnn | sgd | 0.24M | 86.40 ± 4.37 | 86.24 | 86.83 |
| D_opt_rmsprop | IMU | cnn | rmsprop | 0.24M | 86.31 ± 9.32 | 86.20 | 86.83 |
| D_opt_adagrad | IMU | cnn | adagrad | 0.24M | 84.76 ± 8.76 | 84.80 | 85.63 |
| B_fusion3_mlp | IMU+FP+EMG | mlp | adamw | 4.58M | 84.41 ± 9.85 | 84.17 | 84.91 |
| A_EMG_cnn | EMG | cnn | adamw | 0.16M | 71.73 ± 3.70 | 71.70 | 74.56 |
| A_EEG_cnn | EEG | cnn | adamw | 0.16M | 65.66 ± 4.68 | 65.29 | 71.02 |

## Soft-voting ensemble

Members: B_fusion3_cnn, B_fusion3_gru, B_fusion3_lstm, B_fusion3_transformer, C_fusion3_cnn_dae

- Window accuracy: **98.59%** (macro-F1 98.60%)
- Trial accuracy: **98.74%**

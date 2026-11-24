## All runs (5-fold subject-independent CV)

| Run | Modalities | Encoder | Optimizer | Params | Window acc | Window F1 | Trial acc |
|---|---|---|---|---|---|---|---|
| B_resnet_dilated | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 98.01 ± 2.21 | 98.00 | 98.74 |
| C_resnet_dilated_4mod | IMU+FP+EMG+EEG | resnet_dilated | adamw | 2.52M | 97.75 ± 2.86 | 97.73 | 98.10 |
| C_resnet_dilated_dae | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 97.51 ± 1.95 | 97.52 | 98.74 |
| G_resnet_dilated_ganaug | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 97.42 ± 3.10 | 97.41 | 96.86 |
| B_resnet | IMU+FP+EMG | resnet | adamw | 2.07M | 97.07 ± 3.04 | 97.06 | 97.48 |
| B_efficientnet | IMU+FP+EMG | efficientnet | adamw | 1.55M | 96.51 ± 1.65 | 96.47 | 98.74 |
| B_vgg | IMU+FP+EMG | vgg | adamw | 2.89M | 96.50 ± 3.60 | 96.49 | 97.48 |
| C_resnet_dilated_ae | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 96.47 ± 5.95 | 96.44 | 97.48 |
| C_resnet_dilated_vae | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 95.92 ± 3.63 | 95.93 | 96.86 |
| C_resnet_dilated_sae | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 95.87 ± 4.50 | 95.84 | 96.23 |
| B_mobilenet | IMU+FP+EMG | mobilenet | adamw | 1.33M | 95.82 ± 1.32 | 95.82 | 97.48 |
| B_vit | IMU+FP+EMG | vit | adamw | 3.05M | 95.77 ± 3.37 | 95.73 | 97.48 |
| B_gru | IMU+FP+EMG | gru | adamw | 1.46M | 95.63 ± 2.56 | 95.62 | 96.23 |
| B_densenet | IMU+FP+EMG | densenet | adamw | 1.85M | 95.37 ± 3.23 | 95.38 | 96.23 |
| B_bert | IMU+FP+EMG | bert | adamw | 3.28M | 95.09 ± 2.91 | 95.13 | 94.97 |
| B_inception | IMU+FP+EMG | inception | adamw | 0.91M | 95.05 ± 2.86 | 95.07 | 95.60 |
| A_FP_resnet | FP | resnet | adamw | 0.54M | 94.54 ± 1.86 | 94.53 | 94.22 |
| D_opt_adam | IMU | resnet | adam | 0.62M | 93.83 ± 3.27 | 93.85 | 95.21 |
| B_transformer | IMU+FP+EMG | transformer | adamw | 2.82M | 93.66 ± 1.61 | 93.68 | 96.23 |
| D_opt_adamw | IMU | resnet | adamw | 0.62M | 92.65 ± 5.98 | 92.64 | 92.81 |
| B_lstm | IMU+FP+EMG | lstm | adamw | 1.74M | 92.21 ± 3.85 | 92.18 | 93.71 |
| D_opt_adagrad | IMU | resnet | adagrad | 0.62M | 91.90 ± 3.91 | 91.97 | 93.41 |
| A_IMU_resnet | IMU | resnet | adamw | 0.62M | 91.84 ± 6.18 | 91.82 | 92.81 |
| B_lstm_uni | IMU+FP+EMG | lstm_uni | adamw | 0.95M | 91.77 ± 6.37 | 91.74 | 93.08 |
| G_IMU_resnet_ganaug | IMU | resnet | adamw | 0.62M | 91.68 ± 3.25 | 91.66 | 92.81 |
| B_convnext | IMU+FP+EMG | convnext | adamw | 3.54M | 91.03 ± 3.29 | 91.03 | 91.19 |
| D_opt_rmsprop | IMU | resnet | rmsprop | 0.62M | 87.48 ± 7.61 | 87.50 | 88.62 |
| D_opt_nesterov | IMU | resnet | nesterov | 0.62M | 86.73 ± 6.78 | 86.74 | 86.83 |
| D_opt_momentum | IMU | resnet | momentum | 0.62M | 86.11 ± 6.66 | 86.11 | 86.83 |
| D_opt_sgd | IMU | resnet | sgd | 0.62M | 82.67 ± 4.53 | 82.64 | 82.04 |
| B_mlp | IMU+FP+EMG | mlp | adamw | 4.71M | 82.07 ± 11.28 | 81.92 | 85.53 |
| B_mlp_wn | IMU+FP+EMG | mlp_wn | adamw | 4.71M | 77.67 ± 8.38 | 77.04 | 78.62 |
| A_EMG_resnet | EMG | resnet | adamw | 0.54M | 68.12 ± 5.73 | 68.09 | 70.41 |
| B_rnn | IMU+FP+EMG | rnn | adamw | 0.90M | 63.66 ± 9.19 | 62.67 | 64.78 |
| A_EEG_resnet | EEG | resnet | adamw | 0.54M | 52.08 ± 1.73 | 50.52 | 51.70 |
| G_resnet_dilated_TSTR | IMU+FP+EMG | resnet_dilated | adamw | 2.02M | 49.22 ± 9.12 | 46.76 | 49.06 |

## Soft-voting ensemble

Members: B_resnet_dilated, C_resnet_dilated_dae, G_resnet_dilated_ganaug, B_resnet, B_efficientnet

- Window accuracy: **98.68%** (macro-F1 98.68%)
- Trial accuracy: **99.37%**

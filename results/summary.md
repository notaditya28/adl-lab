## All runs (5-fold subject-independent CV)

| Run | Modalities | Encoder | Optimizer | Params | Window acc | Window F1 | Window AUC-ROC | Trial acc |
|---|---|---|---|---|---|---|---|---|
| E_conv_ae | IMU+FP+EMG | conv_ae | adamw | 0.74M | 96.92 ± 1.29 | 96.89 | 99.56 | 97.48 |
| E_sparse_ae | IMU+FP+EMG | resnet | adamw | 2.07M | 96.57 ± 2.47 | 96.56 | 99.80 | 97.48 |
| E_bigan | IMU+FP+EMG | resnet | adamw | 2.07M | 94.81 ± 3.24 | 94.79 | 99.16 | 94.97 |

## Unsupervised Reconstruction & Anomaly Detection (Sparse AE, Conv-AE, BiGAN)

| Run | Model / Pretrain | Modalities | Recon MSE | Recon MAE | Anomaly AUC-ROC | Anomaly F1 | Class AUC-ROC | Class F1 |
|---|---|---|---|---|---|---|---|---|
| E_conv_ae | conv_ae (conv_ae) | IMU+FP+EMG | 1.1216 ± 1.4323 | 0.3906 ± 0.0278 | 99.15 | 98.08 | 99.56 | 96.89 |
| E_sparse_ae | resnet (sae) | IMU+FP+EMG | 1.3794 ± 1.3999 | 0.4832 ± 0.0390 | 99.69 | 96.80 | 99.80 | 96.56 |
| E_bigan | resnet (bigan) | IMU+FP+EMG | 1.5447 ± 1.3962 | 0.5666 ± 0.0380 | 99.98 | 97.97 | 99.16 | 94.79 |

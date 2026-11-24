# Walking-Speed Recognition from Multimodal Gait Signals (ADL Lab Project)

Deep-learning classification of treadmill walking speed (**0.5 / 0.75 / 1.0 m/s**) from the
PhysioNet *Multimodal Gait Dataset* (59 healthy young adults; EEG, EMG, IMU kinematics and
force plates). Every model is evaluated **subject-independently**: the people in each test fold
are never seen during training.

## Headline result

| Model | Window accuracy | Trial accuracy |
|---|---|---|
| **Soft-voting ensemble** (CNN + CNN-DAE + Transformer + BiLSTM + BiGRU, IMU+FP+EMG) | **98.59 %** (macro-F1 98.60 %) | **98.74 %** (157/159 trials) |
| Best single model: ResNet-DS-CNN with denoising-autoencoder pre-training | 97.40 % | 97.48 % |
| Best single model without pre-training: ViT-style Transformer | 96.74 % | 97.48 % |

*Window = one 4-s segment; trial = one 60-s recording (softmax averaged over its windows).*
All errors are between **adjacent** speeds; 0.5 m/s is never mistaken for 1.0 m/s
(`results/confusion_ensemble.png`).

## Data and preprocessing (`src/prepare_data.py`)

| Modality | Raw | After preprocessing | Trials |
|---|---|---|---|
| IMU (Xsens, 17 sensors) | 60 Hz | joint angles + segment angular velocity + sensor free acceleration, 204 ch | 167 |
| Force plates (split-belt) | 1000 Hz | Fx,Fy,Fz,Mx,My,Mz (÷ body weight) + COP per belt, FIR-decimated to 100 Hz, 16 ch | 173 |
| EMG (Delsys, 12 muscles) | ~16 Hz MAV | log-MAV, resampled to 16 Hz, trailing zero padding removed | 169 |
| EEG (DSI-24, dry) | 300 Hz | 18 scalp ch, 1–40 Hz band-pass, common-average reference, 100 Hz, clipped ±150 µV | 176 |

Dataset issues found and handled:
- **S32, EEG at 1.0 m/s:** the channel header is corrupted, so the recording is excluded.
- **S47, IMU:** the joint-angle CSVs are empty at all 3 speeds, so S47 has no IMU data.
- **S2, force plates:** the CSVs have no header row, so they are parsed using the standard 22-column layout.
- **Missing files:** the gaps listed on PhysioNet are simply absent.

Every downloaded file was checked against PhysioNet's `SHA256SUMS.txt`.

## Evaluation protocol (`src/train.py`)

- **5-fold cross-validation grouped by subject** (seed 42). Inside each training fold, 15% of
  subjects are held out for early stopping and checkpoint selection. The test fold is never used
  for any decision during training.
- Normalisation statistics (per-channel z-score) are computed **only on that fold's training
  subjects**.
- 4-s windows with a 1-s hop. Training uses a random temporal crop (jitter inside the hop),
  per-channel amplitude scaling and Gaussian noise.
- Metrics: window accuracy and macro-F1, trial accuracy, and the standard deviation across folds.

## Syllabus concepts and where they are used

| Unit | Concept | Implementation |
|---|---|---|
| ML preliminaries | Supervised learning, bias–variance, under/over-fitting | Subject-grouped CV + validation split; MLP (4.6 M params) overfits (±9.9 % fold std) vs. CNN (0.8 M, ±1.9 %) |
| | Deep feed-forward nets, backprop, SGD | `MLPEncoder`; whole pipeline trained with backprop + mini-batch SGD variants |
| | Activations, initialization, vanishing/exploding grads | ReLU/GELU, He (Kaiming) init, residual connections, gradient-norm clipping (1.0) |
| Optimization | Mini-batch SGD, Momentum, Nesterov, Adagrad, RMSProp, Adam | Optimizer study (phase D) |
| | BatchNorm, LayerNorm | BN in CNN/MLP; pre-LN Transformer, LN in RNN and fusion head |
| | L1/L2, Dropout, Early stopping | AdamW decoupled L2 (wd 1e-2), optional `--l1`, Dropout, modality dropout, early stopping (patience 10), label smoothing 0.1 |
| Advanced CNNs | Depthwise-separable + dilated convs, stride, pooling, ResNet / MobileNet / EfficientNet ideas | `ResCNNEncoder`: residual depthwise-separable blocks, dilation 1–8, strided downsampling, Squeeze-Excitation, global average pooling |
| Sequence models | RNN, bidirectional RNN, many-to-one | `RNNEncoder` (BiGRU / BiLSTM, many-to-one) |
| Attention models | GRU, LSTM, Attention, Transformer, ViT | Additive-attention pooling over RNN states; ViT-style 1-D Transformer (patch embedding, [CLS] token, learned positions); cross-modal self-attention fusion |
| Generative models | Denoising autoencoder | `DenoisingAE` (noise + time-masking corruption) pre-trains each modality's CNN encoder without labels |

## Results (5-fold subject-independent CV)

### A. Single modality (ResNet-DS-CNN)
| Modality | Window acc | Trial acc |
|---|---|---|
| Force plates | 93.29 ± 3.19 | 93.64 |
| IMU | 91.19 ± 4.57 | 92.81 |
| EMG | 71.73 ± 3.70 | 74.56 |
| EEG | 65.66 ± 4.68 | 71.02 |

Chance level is 33.3 %. EEG is well above chance, but during walking much of that signal is
probably motion artifact rather than brain activity.

### B. Architecture comparison (IMU + FP + EMG, attention fusion)
| Encoder | Params | Window acc | Trial acc |
|---|---|---|---|
| Transformer (ViT-1D) | 2.97 M | 96.74 ± 0.66 | 97.48 |
| ResNet-DS-CNN | 0.79 M | 96.39 ± 1.94 | 96.23 |
| BiLSTM + attention | 1.70 M | 95.90 ± 0.94 | 96.86 |
| BiGRU + attention | 1.42 M | 91.63 ± 4.85 | 91.82 |
| MLP | 4.58 M | 84.41 ± 9.85 | 84.91 |

### C. Fusion and pre-training
| Setting | Window acc | Trial acc |
|---|---|---|
| CNN + denoising-AE pre-training (IMU+FP+EMG) | **97.40 ± 3.40** | 97.48 |
| CNN, all 4 modalities (adds EEG) | 95.20 ± 4.44 | 95.57 |

Fusing three modalities beats the best single modality by about 3 points. Adding EEG makes it
slightly *worse*, because the noisy dry-electrode EEG recorded during walking adds variance.
Denoising-AE pre-training adds about 1 point to the CNN.

### D. Optimizer study (IMU, CNN, 25 epochs, OneCycle LR)
| Optimizer | Window acc | Trial acc |
|---|---|---|
| Adam | 91.31 ± 2.94 | 92.81 |
| Nesterov momentum | 89.82 ± 3.99 | 91.62 |
| Momentum | 89.68 ± 4.06 | 91.02 |
| AdamW | 88.97 ± 4.93 | 89.82 |
| SGD (plain) | 86.40 ± 4.37 | 86.83 |
| RMSProp | 86.31 ± 9.32 | 86.83 |
| Adagrad | 84.76 ± 8.76 | 85.63 |

Optimizers that use momentum or a first moment (Adam, Nesterov, Momentum, AdamW) are about 3–5
points better and more stable than plain SGD, RMSProp and Adagrad. Adagrad's shrinking learning
rate hurts most. Differences of about 2 points are inside fold-to-fold noise: AdamW reaches
91.2 % with 40 epochs in phase A.

## Caveats (for honest reporting)
- The ensemble's members were chosen *after* looking at cross-validation scores, so the 98.6 %
  is slightly optimistic. The single-model numbers do not have this problem.
- Speed was always recorded in the same order (0.5 → 0.75 → 1.0), so fatigue or time-on-task
  is confounded with speed. This is a property of the dataset.
- Force-plate data is only approximately synchronised with the other modalities, which matters
  little here because labels are per trial.

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install numpy pandas scipy scikit-learn matplotlib
bash download.sh csv; bash download.sh imu      # ~6.5 GB, checksum-verified
tar xzf data/IMU_Data/IMU.tar.gz -C data/IMU_Data
.venv\Scripts\python src\prepare_data.py
bash run_experiments.sh                          # ~1 h on an RTX 4060 Laptop GPU
```

Single run: `.venv\Scripts\python src\train.py --mods IMU FP EMG --encoder transformer --lr 1e-3`

## Files
- `src/prepare_data.py` – raw CSV → per-trial arrays
- `src/models.py` – MLP, ResNet-DS-CNN, BiLSTM/BiGRU+attention, Transformer, fusion, denoising AE
- `src/train.py` – subject-grouped CV training/evaluation
- `src/report.py` – tables, ensemble, figures
- `results/summary.md`, `results/results.jsonl`, `results/*.png`, `results/preds/*.npz`

## Citation
Katmah R., AlShehhi A., Kosaji D., Al-Rahmani N., Abdullah M., Hulleck A. A. V., Khalaf K. (2026).
*A multimodal gait dataset of brain activity, muscle activity, kinematics and ground forces in
young adults* (v1.0.0). PhysioNet. https://doi.org/10.13026/r0ea-7161 — CC BY 4.0.

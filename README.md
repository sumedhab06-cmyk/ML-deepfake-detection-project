# DeepFake Face Image Detection

A complete mini-project for binary classification of **cropped facial images** as `Real` (label 0) or `Fake` (label 1). It compares MobileNetV3-Small, EfficientNet-B0, and ResNet-50 using ImageNet pretrained weights, a source-video-grouped validation split, and an independent CACD test set. No video detection is implemented.

## What is included

| Archive | Use | Images |
| --- | --- | ---: |
| `celebdf (1).tar.gz` | Train/validation real | 18,108 |
| `celebdf_recons (1).tar.gz` | Train/validation fake (reconstructions) | 18,108 |
| `cacd_auth.tar.gz` | Final test real | 5,063 |
| `cacd_deepfake.tar.gz` | Final test fake (face swaps) | 3,756 |

The original dataset README is in `docs/DATASET_README.md`. The original project synopsis is in `docs/synopsis.docx`. **The synopsis currently names FaceForensics++ as its data source; revise that section before submission because these supplied archives are CelebDF/CACD based.** It also describes unseen dataset generalization; the CACD result is one cross-source test, not a guarantee of broad generalization. The dataset README's identity count does not match the folder counts (57 training folders, 47 test folders).

## Setup

Use Python 3.11 or newer. In a terminal opened inside this folder:

```bash
python -m venv .venv
```

Activate it (`.venv\\Scripts\\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux), then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation varies by GPU and operating system; if the command above fails or you need CUDA acceleration, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for matching `torch` and `torchvision`, then install the other requirements. Pretrained weights download on first training run and require internet access. You can use `--no-pretrained` for an offline code check, but that is **not** the proposed transfer-learning experiment.

## Run

```bash
python prepare.py
python train.py --model mobilenet_v3 --epochs 8 --batch-size 32
python evaluate.py --model mobilenet_v3
```

Repeat `train.py` and `evaluate.py` for `efficientnet_b0` and `resnet50`. Alternatively:

```bash
python run_all.py --epochs 8 --batch-size 32
```

On limited GPU memory, lower `--batch-size` to 8 or 16. CPU training is possible but may be slow. `--device cpu` or `--device cuda` overrides auto detection. Test files are never used to choose checkpoints: `train.py` selects the best epoch by validation ROC-AUC; `evaluate.py` then measures CACD performance. Run `evaluate.py` after training decisions are final. Do not adjust the model or decision threshold based on the final test results and describe that same test as untouched.

To predict an **already cropped face** after training:

```bash
python predict.py --image path/to/cropped_face.jpg --checkpoint outputs/mobilenet_v3/best.pt
```

## Deploy a web demo

First train and evaluate a model. Pick one model using its validation results, then run:

```bash
python package_deployment.py --model mobilenet_v3
```

This creates `DeepFake_Deployment_mobilenet_v3.zip` with a **standalone Streamlit app and the trained checkpoint**. Extract that ZIP and run it locally:

```bash
cd deployment_ready
python -m pip install -r requirements.txt
streamlit run app.py
```

To publish, upload the **extracted deployment ZIP contents**, including `model/best.pt`, to the root of a new GitHub repository. Go to [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select your repository and branch, set the entrypoint to `app.py`, choose Python 3.11 or 3.12 in Advanced settings, and deploy. The hosted demo is CPU only by default and may be slower than local GPU inference. Do **not** upload the four training archives or extracted images to the deployment repository. The app handles cropped face uploads in memory and does not deliberately persist them. Check the service's privacy terms before uploading personal photos.

Outputs for each model appear in `outputs/<model>/`: `best.pt`, `history.csv`, `test_metrics.json`, `confusion_matrix.png`, and `roc_curve.png`. Report accuracy, specificity, fake-class precision/recall/F1, ROC-AUC, confusion matrix, model complexity (parameter count, if desired), and batch-amortized inference time. The included time measure includes image loading/preprocessing and device transfers; it is **not** single-image latency. Compare timings only on the same hardware and batch size.

## Split and interpretation

`prepare.py` extracts the archives once and writes `data/splits.csv`. The five source videos for each training identity are split at the **video** level (one for validation, four for training), with each original image and its matching reconstruction assigned together. All CACD images stay in the test split. Validation is balanced between real and fake because train frames have a paired reconstruction; CACD is mildly imbalanced, so inspect precision, recall, F1, AUC, and the confusion matrix as well as accuracy.

This dataset tests reconstruction-trained models against face swaps from another image source. It does not establish robustness to arbitrary generators, diffusion images, unseen identities, or uncropped photographs. The supplied test identities overlap the training identities (47 of 57), so this is a **cross-source/manipulation** evaluation, not an unseen-identity evaluation. Face crops are low resolution; upscaling to 224 pixels does not add detail. The originals and reconstructions may have aligned content, so group membership and paired examples must remain together. Check your institution's usage and citation requirements for the dataset and paper in `docs/DATASET_README.md`.

## Files

- `prepare.py`: safe archive extraction and split manifest
- `train.py`: fine-tuning with early stopping and best checkpoint
- `evaluate.py`: held-out test metrics and plots
- `predict.py`: a single cropped face prediction
- `run_all.py`: sequential run for all three models
- `deepfake/`: datasets, architectures, and evaluation helpers

Training checkpoints and generated results are **not included**. They require you to run training on your own machine. The original four image archives are included in `data/archives/`.

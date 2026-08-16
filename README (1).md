# 🧠 Alzheimer's MRI Classification — Complete Pipeline

A single Colab notebook covering the full workflow for Alzheimer's MRI analysis: 4-class
classification with **DenseNet169**, explainability via **EfficientNetB0 + Grad-CAM**, and
**U-Net** hippocampal segmentation with clinical-style overlays.

> ⚡ Run on a **T4 GPU**: Runtime → Change runtime type → T4 GPU

## Pipeline overview

| Step | Description |
|------|-------------|
| 1 | **Dataset** — auto-downloaded from HuggingFace (`Falah/Alzheimer_MRI`), no Kaggle/Drive login needed |
| 2 | Imports & global config |
| 3 | Train / val / test split (70 / 15 / 15) |
| 4 | Data generators + class distribution plot |
| 5 | Sample image grid |
| 6 | **DenseNet169** model build |
| 7 | Phase 1 training (frozen backbone) |
| 8 | Phase 2 fine-tuning (last 40 layers unfrozen) |
| 9 | Training curves |
| 10 | Test evaluation — accuracy, AUC, confusion matrix, ROC |
| 11 | **EfficientNetB0** + Grad-CAM heatmaps |
| 12 | **U-Net** hippocampal segmentation |
| 13 | Clinical hippocampal overlay visualizations |
| 14 | Full pipeline on any single image |
| 15 | Upload & predict your own MRI |

## Dataset

[`Falah/Alzheimer_MRI`](https://huggingface.co/datasets/Falah/Alzheimer_MRI) on HuggingFace — 4 classes:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

The notebook downloads and splits the dataset automatically; no manual setup required.

## Models

- **DenseNet169** (ImageNet-pretrained) — primary classifier, two-phase training (frozen backbone → fine-tuned last 40 layers)
- **EfficientNetB0** — secondary classifier used for Grad-CAM explainability heatmaps
- **U-Net** (from scratch) — hippocampal region segmentation on grayscale MRI slices, feeding into a clinical-style red overlay visualization

## Getting started

1. Open `Alzheimer_Final_v4.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set the runtime to a T4 GPU
3. Run all cells top to bottom — the dataset, splitting, training, and evaluation are fully automated
4. Use the final cell to upload your own MRI image and run it through the full pipeline (classification + Grad-CAM + segmentation)

### Local setup (alternative to Colab)

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook Alzheimer_Final_v4.ipynb
```

A GPU is strongly recommended for training; inference-only use of a pretrained checkpoint is feasible on CPU.

## Repository structure

```
.
├── Alzheimer_Final_v4.ipynb   # main pipeline notebook
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

See `requirements.txt`. Core dependencies: TensorFlow/Keras, scikit-learn, OpenCV, pandas,
matplotlib, seaborn, and the HuggingFace `datasets` library.

## Notes & limitations

- This is a research/educational pipeline, **not a diagnostic tool** — it is not validated for
  clinical use.
- The hippocampal segmentation U-Net is trained without ground-truth anatomical masks in this
  version; treat its output as an illustrative region-of-interest map rather than a clinically
  validated segmentation.
- Model weights (`best_weights.keras`) and the downloaded dataset are not tracked in this repo
  (see `.gitignore`) — they are regenerated on first run.

## License

Add a license of your choice (e.g. MIT) — see [choosealicense.com](https://choosealicense.com).

## Acknowledgments

- Dataset: [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) on HuggingFace
- Pretrained backbones: `DenseNet169` and `EfficientNetB0` (ImageNet weights) via `tf.keras.applications`

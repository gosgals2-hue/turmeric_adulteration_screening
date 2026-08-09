# Turmeric_adulteration_screening
Tiered ML pipeline (logistic regression, SVM, MobileNetV2) with leakage-free evaluation and a photometric degradation study to screen for turmeric adulteration.
Approach

A tiered pipeline of increasing capacity, each testing what kind of colour signal separates pure from adulterated turmeric:
Tier 1: logistic regression on 11 colour/texture statistics (average colour)
Tier 2: RBF support vector machine on a 96-bin colour histogram (colour distribution)
Tier 3: MobileNetV2 CNN via transfer learning (learned deep features)

Repo structure:

src/                 # pipeline scripts
  segment_bg.py            # illumination-independent segmentation
  extract_frames.py        # video → frames
  extract_features.py      # 11 colour/texture stats (Tier 1)
  extract_hist_features.py # 96-bin colour histogram (Tier 2)
  split_dataset.py         # pile-aware train/val/test split
  train_tier1_logreg.py    # Tier 1
  train_tier2_svm.py       # Tier 2 (GroupKFold-tuned)
  train_tier3_cnn.py       # Tier 3 (MobileNetV2, PyTorch)
  evaluate_protocol.py     # metrics: AUROC, per-ratio sensitivity, prevalence adjustment
  degrade_eval.py          # degradation / robustness study
report/              # write-up, figures
data/               #included when able to fit large image and video files

Running it:

Requires Python 3, with OpenCV, NumPy, pandas, scikit-learn, PyTorch + torchvision, and Matplotlib. The pipeline is deterministic (fixed seed 42) and reproducible from the raw data: each stage reads data/metadata.csv and data/split/manifest.csv. Typical order: split_dataset.py → extract_* → train_tier* → evaluate_protocol.py → degrade_eval.py.

Author: Prisha Goswami. Created for Non-Trivial Fellowship, 2026.

# PHAS0077 — Explainability and Uncertainty in Chest X-ray Classification

This repository contains the code for the MSc project:

**Explainability and Uncertainty in Chest X-ray Classification:  
A Regional Occlusion Study of Model Reliability**

The project investigates the reliability of a four-class chest X-ray classifier by
combining model explanation, predictive uncertainty, probability calibration and
controlled regional occlusion.

## Main Entry Point

The primary submission file is:

`notebooks/final_experiment_pipeline.ipynb`

This is the final pipeline used for the dissertation-scale experiment. The notebook
has been executed on the full 100-image cohort, and the saved outputs correspond to
the results reported in the dissertation.

Earlier notebooks are retained in `development_history/` to document the progression
of the project. They are not required to reproduce the final analysis.

## Scientific Workflow

The final notebook:

1. builds a single-channel ImageNet-pretrained ResNet-18 baseline;
2. uses a fixed stratified 70:15:15 train/validation/test split with seed 42;
3. handles class imbalance using inverse-frequency weighted random sampling;
4. constructs and fine-tunes a ResNet-18 with dropout before the final classifier;
5. fits post-hoc Temperature Scaling on the validation split;
6. evaluates the baseline, MC Dropout and Temperature Scaling configurations on the
   frozen test set;
7. selects a final 100-image cohort using a composite uncertainty score based on
   predictive entropy, inverse confidence and inverse Top-1–Top-2 probability margin;
8. applies a fixed 7 × 7 regional occlusion procedure under all three configurations;
9. checks MC Dropout stability using 10, 20, 30 and 50 stochastic forward passes;
10. generates baseline Score-CAM maps and computes Hit@5, IoU@5, Top-1 agreement,
    Manhattan distance and grid-level rank correlation;
11. exports regional-response tables, abnormal-case summaries, overlap statistics and
    visualisations.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── final_experiment_pipeline.ipynb
│
├── development_history/
│   ├── 01_baseline_training_and_evaluation.ipynb
│   ├── 02_gradcam_scorecam_development.ipynb
│   ├── 03_peripheral_shortcut_masking.ipynb
│   ├── 04_scorecam_and_regional_occlusion_exploration.ipynb
│   └── 05_three_configuration_uq_development.ipynb
│
├── src/
│   ├── metrics.py
│   └── occlusion.py
│
├── scripts/
│   └── quick_reproducibility_check.py
│
├── tests/
│   └── test_core_metrics.py
│
├── sample_data/
│   └── sample_probabilities.csv
│
└── results/
    ├── expected_quick_check.json
    ├── reference_model_metrics.csv
    ├── reference_regional_metrics.csv
    └── reference_overlap_metrics.csv
```

## Dataset

The experiments use the **COVID-19 Radiography Database**, available from Kaggle:

[COVID-19 Radiography Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

The version used in this project contains **21,165 chest X-ray images** across four
classes:

- COVID: 3,616 images
- Lung Opacity: 6,012 images
- Normal: 10,192 images
- Viral Pneumonia: 1,345 images

The complete image dataset is not bundled in this repository because of its size.
Please download and extract the dataset before running the full experimental pipeline.

The expected directory structure is:

```text
COVID-19_Radiography_Dataset/
├── COVID/
│   └── images/
├── Lung_Opacity/
│   └── images/
├── Normal/
│   └── images/
└── Viral Pneumonia/
    └── images/
```

For local execution, set the dataset location using the environment variable:

```bash
export CXR_DATA_ROOT=/path/to/COVID-19_Radiography_Dataset
```

For Google Colab, the notebook also supports loading a dataset archive from Google
Drive. The data-setup section of the notebook explains the relevant configuration.

## Environment

The final pipeline was developed in Python using PyTorch and was executed in
Google Colab with CUDA acceleration.

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

A GPU-enabled environment is strongly recommended for the full experiment because
MC Dropout, Score-CAM and regional occlusion require repeated model evaluations.

## Quick Reproducibility Check

A lightweight reproducibility check is included so that the core probability and
uncertainty calculations can be tested without downloading the complete chest X-ray
dataset.

Run:

```bash
python scripts/quick_reproducibility_check.py
```

The script loads:

```text
sample_data/sample_probabilities.csv
```

and recomputes:

- prediction confidence;
- predictive entropy;
- Top-1–Top-2 probability margin;
- multiclass Brier score;
- negative log-likelihood.

The calculated values are checked against:

```text
results/expected_quick_check.json
```

The supplied probability table is a lightweight synthetic test input used only to verify
the core metric calculations. It is not part of the medical dataset and is not used in
the dissertation analysis.

Optional unit tests can also be run with:

```bash
pytest -q
```

## Running the Final Experiment

Open:

```text
notebooks/final_experiment_pipeline.ipynb
```

configure the dataset location, and run the notebook from top to bottom.

For a fast pipeline check before running the complete experiment, set:

```python
MAX_TARGET_SAMPLES = 5
```

For the full dissertation-scale experiment, use:

```python
TARGET_SAMPLE_COUNT = 100
MAX_TARGET_SAMPLES = None

GRID_SIZE = 7

N_MC_EVAL = 30
N_MC_OCCLUSION = 30

RUN_FULL_OCCLUSION = True
RUN_MC_STABILITY_CHECK = True
RUN_SCORECAM_OVERLAP = True
```

The final controlled regional experiment evaluates:

```text
100 images × 49 masked regions × 3 model configurations
= 14,700 masked-region evaluations
```

The notebook supports checkpoint reuse and resumable regional occlusion and
Score-CAM processing. Run signatures incorporate checkpoint, split and cohort
information to reduce the risk of incompatible partial results being combined.

## Main Reference Results

Compact reference summaries from the final experiment are provided in the
`results/` directory.

### Test-set Performance

| Configuration | Accuracy | Macro F1 | Macro AUC | ECE | NLL | Brier Score |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ResNet-18 | 0.9443 | 0.9496 | 0.9931 | 0.0205 | 0.1630 | 0.0872 |
| MC Dropout | 0.9502 | 0.9571 | 0.9937 | 0.0144 | 0.1534 | 0.0797 |
| Temperature Scaling | 0.9443 | 0.9496 | 0.9931 | 0.0101 | 0.1558 | 0.0857 |

MC Dropout achieved the highest test accuracy, while Temperature Scaling produced
the lowest Expected Calibration Error without changing the baseline predicted classes.

### Regional Occlusion

The final regional experiment produced **14,700 masked-region evaluations**.

The predicted-class change rates were:

- Baseline ResNet-18: 20.31%
- MC Dropout: 8.24%
- Temperature Scaling: 20.31%

MC Dropout produced the largest mean absolute entropy response despite showing the
lowest predicted-class change rate.

### Score-CAM and Uncertainty-sensitive Region Overlap

Hit@5 measured whether the single most uncertainty-sensitive region was contained
within the five highest-scoring Score-CAM regions.

The final Hit@5 values were:

- Baseline ResNet-18: 42%
- MC Dropout: 25%
- Temperature Scaling: 42%

The limited overlap indicates that prediction-supporting regions and
uncertainty-sensitive regions provide related but distinct information about model
behaviour.

The files in `results/` are provided as compact reference summaries rather than as
replacements for rerunning the complete experimental pipeline.

## Reproducibility Settings

- Global random seed: 42
- Dataset split: 70% training, 15% validation, 15% test
- Split type: fixed stratified split
- Image size: 224 × 224
- Input channels: 1
- Training batch size: 32
- Baseline model: ImageNet-pretrained ResNet-18
- Baseline training: 5 epochs
- Optimiser: Adam
- Initial learning rate: 1e-4
- Class imbalance treatment: inverse-frequency weighted random sampling
- MC Dropout probability: 0.30
- MC Dropout fine-tuning: 5 additional epochs
- Final MC Dropout inference: 30 stochastic passes
- BatchNorm during MC inference: evaluation mode
- Temperature Scaling: fitted only on validation logits
- Final cohort size: 100 images
- Composite uncertainty weighting:
  - predictive entropy: 0.50
  - inverse confidence: 0.25
  - inverse Top-1–Top-2 probability margin: 0.25
- Regional analysis: 7 × 7 grid
- Regions per image: 49
- Grid-cell size: 32 × 32 pixels
- Mask value: 0 in normalised tensor space
- Total final masked-region evaluations: 14,700

## Interpretation Note

Grad-CAM and Score-CAM outputs are treated as model-attribution maps rather than
clinically verified lesion localisation.

Similarly, regional occlusion measures the model's response to a defined input
perturbation. Changes in predicted class, confidence or predictive entropy should not
be interpreted as causal evidence that a masked region contains pathological
information.

The analyses are intended as tools for auditing model behaviour and regional
sensitivity rather than for establishing clinical localisation or diagnostic causality.

## Author

**Hao Chen**  
MSc Scientific and Data Intensive Computing  
Department of Physics and Astronomy  
University College London
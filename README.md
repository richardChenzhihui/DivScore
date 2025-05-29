# DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains


This repository contains the official implementation for the paper:  
**DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains**

[Anonymous ACL Submission] | [PDF](assests/DivScore_Zero_Shot_Detect.pdf)

## 📝 Abstract
Detecting LLM-generated text in specialized domains like medicine and law is crucial for combating misinformation. DivScore is a novel zero-shot detection framework that uses normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. Our method outperforms state-of-the-art detectors by 14.4% AUROC and 64.0% recall at 0.1% FPR threshold.

## 🚀 Features
- Zero-shot detection framework for specialized domains
- Domain knowledge distillation via LLM fine-tuning
- Normalized entropy-based scoring mechanism
- Robust against adversarial attacks
- Comprehensive evaluation on medical and legal datasets
- Support for multiple LLM architectures

## ⚙️ Installation

### System Requirements
- CUDA 12.1
- PyTorch 2.3.0
- Python 3.8+

### Create Conda Environment
```bash
conda create -n divscore python=3.8
conda activate divscore
```

### Install Dependencies
```bash
pip install transformers datasets peft trl accelerate scikit-learn pandas tqdm
```

## 🧪 Code Structure

```
DivScore/
├── divscore/                  # Core detection module
│   ├── __init__.py            # Package initialization
│   ├── detector.py            # DivScore detection class
│   ├── metrics.py             # Entropy and cross-entropy calculations
│   └── utils.py               # Utility functions
│
├── experiments/               # Experiment scripts
│   ├── domain_distillation/   # Domain adaptation
│   │   ├── distillation.py    # LoRA fine-tuning script
│   │   ├── load_save_model.py # Model merging and testing
│   │   └── .zip               # Adapted Mistral-7B LoRA weights
│   └── main.py                # Main evaluation pipeline
│
├── datasets/                  # Dataset storage
│   ├── core/                  # Core LLM-generated text detection evaluation datasets
│   │   ├── legal/             # Legal text datasets
│   │   └── medical/           # Medical text datasets
│   ├── domain_distillation/   # Distillation datasets
│   │   ├── legal/             # Legal domian distillation
│   │   └── medical/           # Medical domian distillation
│   │   └── combined/          # Combined (Medical & Legal) distillation
│   └── robustness/            # Adversarial robustness datasets
│
├── assests/                   # Evaluation assets
│   ├── auroc_curve.pdf        # AUROC performance curves
│   ├── binoculars_fastdetect_score.pdf # Baseline comparisons
│   └── ...                    # Additional evaluation assets
│
└── README.md                  # This documentation
```

## 🧠 Domain Knowledge Distillation

To adapt the detector to specialized domains:

1. **Prepare distillation data** (medical/legal QA pairs)
2. **Run distillation script**:
```bash
python experiments/domain_distillation/distillation.py
```

3. **Merge and save adapted model**:
```bash
python experiments/domain_distillation/load_save_model.py
```
Note, we provide adapted Mistral-7B's LoRA weights through Medical, Legal domain and their combination for replicating experiments.

Configuration options in `distillation.py`:
```python
EXPERIMENT_CONFIG = {
    "data_path": "path/to/distillation_data.json",
    "base_model": "path/to/Mistral-7B-v0.2",
    "output_dir": "output/path",
    "seed": 42,
    "lora_rank": 64,
    "lora_alpha": 128,
    "epoch": 8  # Optimized training epochs
}
```

## 🔍 Detection with DivScore

### Basic Usage
```python
from divscore import DivScore

# Initialize detector
detector = DivScore(
    generalLM_name_or_path="path/to/base_model",
    enhancedLM_name_or_path="path/to/domain_adapted_model",
    device="cuda:0"
)

# Compute detection score
# Suggest classifying score < 0.15 as LLM-generated 
text = "Your input text here..."
score, _, _ = detector.compute_score(text)
```

### Full Evaluation Pipeline
Run comprehensive evaluation on medical and legal datasets:
```bash
python experiments/main.py
```

[//]: # ()
[//]: # (## 📊 Results)

[//]: # ()
[//]: # (### Detection Performance &#40;AUROC %&#41;)

[//]: # (| Method          | MIMIC | PubMedQA | OALC | LawStack | Avg. |)

[//]: # (|-----------------|-------|----------|------|----------|------|)

[//]: # (| RoBERTa-base    | 66.79 | 85.79    | 58.33| 65.10    | 69.00|)

[//]: # (| RoBERTa-large   | 93.05 | 91.69    | 81.23| 72.36    | 84.58|)

[//]: # (| Entropy         | 68.52 | 98.29    | 33.30| 91.85    | 72.99|)

[//]: # (| Rank            | 71.97 | 87.54    | 29.97| 85.73    | 68.80|)

[//]: # (| Log-Rank        | 83.37 | 98.75    | 32.40| 91.19    | 76.43|)

[//]: # (| Log-Likelihood  | 81.29 | 98.30    | 32.32| 90.15    | 75.52|)

[//]: # (| DetectGPT       | 53.71 | 42.39    | 31.05| 57.51    | 46.17|)

[//]: # (| Fast-DetectGPT  | 96.24 | 87.77    | 46.32| 69.19    | 74.88|)

[//]: # (| Binoculars      | 98.34 | 95.42    | 38.47| 95.08    | 81.83|)

[//]: # (| **DivScore**    | **99.88** | **99.64** | **98.88** | **99.84** | **99.56** |)

[//]: # ()
[//]: # (### Robustness Against Attacks &#40;AUROC %&#41;)

[//]: # (| Attack          | DivScore | Binoculars |)

[//]: # (|-----------------|----------|------------|)

[//]: # (| None            | 99.88    | 98.34      |)

[//]: # (| Paraphrase      | 93.85    | 85.74      |)

[//]: # (| Word Substitute | 95.21    | 75.02      |)

[//]: # ()
[//]: # (### Key Improvements Over Binoculars)

[//]: # (- Average AUROC improvement: +14.4%)

[//]: # (- Recall at 0.1% FPR: +64.0%)

[//]: # (- Robustness against attacks: +22.8% AUROC advantage)

[//]: # ()
[//]: # ()
[//]: # (![AUROC Curves]&#40;assests/auroc_curve.png&#41;)

## 📚 Citation
```bibtex
@article{divscore2025,
  title={DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains},
  author={Anonymous},
  journal={Submitted to ACL},
  year={2025}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

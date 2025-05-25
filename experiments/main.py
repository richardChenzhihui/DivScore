#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Text Detection Experiment

This script evaluates the performance of Divscore detector on various datasets.
Portions of evaluation metrics calculation adapted from sklearn documentation examples.

License: MIT License
Copyright (c) 2023  Zhihui Chen
"""
import sys
import os
from divscore import DivScore
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from tqdm import tqdm
import pandas as pd
import random
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray,
                      adjusted_scores: np.ndarray,
                      target_fpr: float) -> Dict[str, float]:
    """Calculate metrics at specific FPR threshold.

    Args:
        y_true: Ground truth labels
        adjusted_scores: Detection scores
        target_fpr: Desired false positive rate

    Returns:
        Dictionary containing metrics at specified FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, adjusted_scores, pos_label=1)

    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[-1]
        actual_fpr = fpr[best_idx]
    else:
        best_idx = np.argmin(np.abs(fpr - target_fpr))
        actual_fpr = fpr[best_idx]

    threshold = thresholds[best_idx]
    y_pred = (adjusted_scores >= threshold).astype(int)

    return {
        "tpr": tpr[best_idx],
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "threshold": threshold,
        "actual_fpr": actual_fpr
    }


def run_detection_experiment(distill: str, epoch: int) -> None:
    """Run detection experiment for given distillation type.

    Args:
        distill: Type of distillation ('med_distill', 'legal_distill', etc.)
    """
    your_path = "..."
    detector = DivScore(
        generalLM_name_or_path= f"{your_path}/DivScore/models/Mistral-7B-v0.2/",
        enhancedLM_name_or_path= f"{your_path}/DivScore/experiments/domain_distillation/{distill}/models/Mistral-7B-v0.2-{distill}-epoch{epoch}",
        device='cuda:0'
    )
    dataset_paths = {
        "dsr1_mimic_discharge": f"{your_path}/DivScore/datasets/core/medical/dsr1_mimic_discharge.json",
        "dsr1_pubmedqa": f"{your_path}/DivScore/datasets/core/medical/dsr1_pubmedqa.json",
        "dsr1_LawStack": f"{your_path}/DivScore/datasets/core/legal/dsr1_LawStack.json",
        "dsr1_OALC": f"{your_path}/DivScore/datasets/core/legal/dsr1_OALC.json",
        "dsv3_mimic_discharge": f"{your_path}/DivScore/datasets/core/medical/dsv3_mimic_discharge.json",
        "dsv3_pubmedqa": f"{your_path}/DivScore/datasets/core/medical/dsv3_pubmedqa.json",
        "dsv3_LawStack": f"{your_path}/DivScore/datasets/core/legal/dsv3_LawStack.json",
        "dsv3_OALC": f"{your_path}/DivScore/datasets/core/legal/dsv3_OALC.json",
        "gpt4o_mimic_discharge": f"{your_path}/DivScore/datasets/core/medical/gpt4o_mimic_discharge.json",
        "gpt4o_pubmedqa": f"{your_path}/DivScore/datasets/core/medical/gpt4o_pubmedqa.json",
        "gpt4o_LawStack": f"{your_path}/DivScore/datasets/core/legal/gpt4o_LawStack.json",
        "gpt4o_OALC": f"{your_path}/DivScore/datasets/core/legal/gpt4o_OALC.json",
        "gpto3mini_mimic_discharge": f"{your_path}/DivScore/datasets/core/medical/gpto3mini_mimic_discharge.json",
        "gpto3mini_pubmedqa": f"{your_path}/DivScore/datasets/core/medical/gpto3mini_pubmedqa.json",
        "gpto3mini_LawStack": f"{your_path}/DivScore/datasets/core/legal/gpto3mini_LawStack.json",
        "gpto3mini_OALC": f"{your_path}/DivScore/datasets/core/legal/gpto3mini_OALC.json",
    }

    results = {}
    random.seed(82)

    for dataset_name, json_path in dataset_paths.items():
        metrics = {
            'y_true': [],
            'scores': [],
            'detectors': {
                'd1': {'scores': [], 'entropy': [], 'ce': [], }
            }
        }

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # select 1000 paired human-LLM samples from each dataset
        sample_indices = random.sample(range(len(data)), 1000)
        sampled_data = [data[i] for i in sample_indices]

        print(f"\nProcessing {dataset_name}...")

        for line in tqdm(sampled_data, desc=f"Processing {dataset_name}"):
            samples = [
                (line['llm'], 1, 'llm'),
                (line['human'], 0, 'hum')
            ]

            for text, label, _ in samples:
                score, entropy, ce = detector.compute_score(text)
                metrics['y_true'].append(label)
                metrics['scores'].append(score)

                metrics['detectors']['d1']['scores'].append(score)
                metrics['detectors']['d1']['entropy'].append(entropy)
                metrics['detectors']['d1']['ce'].append(ce)
        y_true = np.array(metrics['y_true'])
        scores = np.array(metrics['scores'])
        adjusted_scores = -scores  # Flip score direction

        # Calculate ROC metrics
        roc_auc = roc_auc_score(y_true, adjusted_scores)
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, adjusted_scores, pos_label=1)
        pr_auc = auc(recall_pr, precision_pr)

        # Store base metrics
        results[dataset_name] = {
            "auroc": roc_auc,
            "pr_auc": pr_auc
        }

        # Add FPR-specific metrics
        fpr_levels = {
            "0.01%": 0.01 / 100,
            "0.1%": 0.1 / 100,
            "1%": 1 / 100
        }

        for level_name, fpr_value in fpr_levels.items():
            metrics_data = calculate_metrics(y_true, adjusted_scores, fpr_value)
            results[dataset_name].update({
                f"tpr@{level_name}fpr": metrics_data["tpr"],
                f"precision@{level_name}fpr": metrics_data["precision"],
                f"recall@{level_name}fpr": metrics_data["recall"],
                f"threshold@{level_name}fpr": metrics_data["threshold"],
                f"actual_fpr@{level_name}": metrics_data["actual_fpr"]
            })

        # Save detailed results
        df = pd.DataFrame({
            'true_label': metrics['y_true'],
            'score': metrics['scores'],
            'd1_score': metrics['detectors']['d1']['scores'],
            'd1_entropy': metrics['detectors']['d1']['entropy'],
            'd1_ce': metrics['detectors']['d1']['ce'],
        })
        os.makedirs(f"./results/{distill}/", exist_ok=True)
        df.to_csv(f"./results//{distill}/{dataset_name}.csv", index=False)

    # Save final metrics
    with open(f"./results//{distill}/final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    for distill in ["med_distill", "legal_distill", "combined_distill"]:
        epoch = 8 # optimized epoch number
        run_detection_experiment(distill,epoch)
    print("Experiment completed successfully")

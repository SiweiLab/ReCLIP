#!/usr/bin/env python3
"""Ten-fold XGBoost evaluation for PTM SWING embeddings."""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class PTMCrossValidator:
    """Ten-fold cross-validator for PTM SWING embeddings."""

    def __init__(self, embedding_file: str, fold_dir: str):
        self.embedding_file = embedding_file
        self.fold_dir = fold_dir
        self.embedding_data: pd.DataFrame | None = None
        self.embedding_mapping: Dict[Tuple[object, ...], object] = {}
        self.results: List[Dict[str, object]] = []

        self.xgb_params = {
            "n_estimators": 375,
            "max_depth": 6,
            "learning_rate": 0.08966,
            "random_state": 42,
        }

    def load_embeddings(self) -> None:
        """Load embedding vectors and create a lookup table."""
        print("Loading embedding data...")
        try:
            self.embedding_data = pd.read_pickle(self.embedding_file)
            print(f"Loaded {len(self.embedding_data)} embedding rows")

            print("Creating embedding lookup...")
            for idx, row in self.embedding_data.iterrows():
                key = (
                    row["Uniprot"],
                    row["PTM"],
                    row["Site"],
                    row["AA"],
                    row["Sequence window(-5,+5)"],
                    row["Int_uniprot"],
                )

                if "Vectors" in row:
                    self.embedding_mapping[key] = row["Vectors"]
                    continue

                for col in ["Combined_Vectors", "embedding", "embeddings", "TransformerVectors"]:
                    if col in row:
                        self.embedding_mapping[key] = row[col]
                        break
                else:
                    print(f"[warn] Missing embedding column; skipped row {idx}")

            print(f"Created {len(self.embedding_mapping)} embedding lookup entries")

        except Exception as exc:
            print(f"Failed to load embedding data: {exc}")
            raise

    @staticmethod
    def calculate_additional_metrics(y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Compute balanced accuracy, Youden's J, t-test, and confusion counts."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youdens_j = sensitivity + specificity - 1

        pos_probs = y_pred_proba[y_true == 1]
        neg_probs = y_pred_proba[y_true == 0]
        if len(pos_probs) > 1 and len(neg_probs) > 1:
            t_stat, p_value = stats.ttest_ind(pos_probs, neg_probs)
        else:
            t_stat, p_value = np.nan, np.nan

        return {
            "balanced_accuracy": balanced_acc,
            "youdens_j": youdens_j,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "t_test_t": t_stat,
            "p_value": p_value,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    def get_features_and_labels(self, fold_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a fold CSV and map rows to embedding vectors and binary labels."""
        fold_data = pd.read_csv(fold_file)

        features = []
        labels = []
        missing_count = 0

        for idx, row in fold_data.iterrows():
            key = (
                row["Uniprot"],
                row["PTM"],
                row["Site"],
                row["AA"],
                row["Sequence window(-5,+5)"],
                row["Int_uniprot"],
            )

            if key not in self.embedding_mapping:
                missing_count += 1
                continue

            features.append(self.embedding_mapping[key])
            effect = str(row["Effect"]).lower()
            if effect == "enhance":
                labels.append(1)
            elif effect == "inhibit":
                labels.append(0)
            else:
                print(f"[warn] Unknown Effect value {row['Effect']!r} at row {idx}")
                missing_count += 1

        if missing_count > 0:
            print(f"[warn] {missing_count} rows had no matching embedding or label")

        return np.array(features), np.array(labels)

    def train_and_evaluate_fold(self, fold_num: int) -> Dict[str, object]:
        """Train and evaluate one fold."""
        print(f"\n=== Fold {fold_num} ===")

        train_file = os.path.join(self.fold_dir, f"fold_{fold_num}_train.csv")
        test_file = os.path.join(self.fold_dir, f"fold_{fold_num}_test.csv")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training fold file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test fold file not found: {test_file}")

        print("Loading training data...")
        X_train, y_train = self.get_features_and_labels(train_file)
        print(f"Train: {len(X_train)} samples, labels 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")

        print("Loading test data...")
        X_test, y_test = self.get_features_and_labels(test_file)
        print(f"Test: {len(X_test)} samples, labels 0={np.sum(y_test == 0)}, 1={np.sum(y_test == 1)}")

        print("Training XGBoost model...")
        clf = XGBClassifier(**self.xgb_params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        results: Dict[str, object] = {
            "fold": fold_num,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_pred_proba),
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        results.update(self.calculate_additional_metrics(y_test, y_pred, y_pred_proba))

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        results["fpr"] = fpr
        results["tpr"] = tpr
        results["precision_curve"] = precision_curve
        results["recall_curve"] = recall_curve
        results["pr_auc"] = auc(recall_curve, precision_curve)

        print(f"Fold {fold_num} results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Balanced accuracy: {results['balanced_accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  PR-AUC: {results['pr_auc']:.4f}")
        print(f"  Youden's J: {results['youdens_j']:.4f}")
        print(f"  t-statistic: {results['t_test_t']:.4f}")
        print(f"  p-value: {results['p_value']:.4f}")

        return results

    def run_cross_validation(self) -> List[Dict[str, object]]:
        """Run ten-fold cross-validation."""
        print("Starting ten-fold cross-validation...")
        self.load_embeddings()

        self.results = []
        for fold_num in range(10):
            try:
                self.results.append(self.train_and_evaluate_fold(fold_num))
            except Exception as exc:
                print(f"[warn] Fold {fold_num} failed: {exc}")

        self.calculate_summary_metrics()
        return self.results

    def calculate_summary_metrics(self) -> None:
        """Print summary metrics across folds."""
        if not self.results:
            print("No valid fold results to summarize.")
            return

        print("\n" + "=" * 50)
        print("Ten-fold cross-validation summary")
        print("=" * 50)

        metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
            "pr_auc",
            "youdens_j",
            "t_test_t",
            "p_value",
        ]

        for metric in metrics:
            values = [result[metric] for result in self.results if metric in result]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric.upper():>10}: {mean_val:.4f} +/- {std_val:.4f}")

        total_train = sum(result["n_train"] for result in self.results)
        total_test = sum(result["n_test"] for result in self.results)
        print(f"{'SAMPLES':>10}: train={total_train}, test={total_test}")

        print("\nPer-fold results:")
        print("Fold  Accuracy  Bal_Acc  Precision  Recall  Specific  F1      AUC     PR-AUC  Youden_J  t_stat   p_value")
        print("-" * 108)
        for result in self.results:
            print(
                f"{result['fold']:4d}  "
                f"{result['accuracy']:8.4f}  "
                f"{result['balanced_accuracy']:7.4f}  "
                f"{result['precision']:9.4f}  "
                f"{result['recall']:6.4f}  "
                f"{result['specificity']:8.4f}  "
                f"{result['f1']:6.4f}  "
                f"{result['auc']:6.4f}  "
                f"{result['pr_auc']:6.4f}  "
                f"{result['youdens_j']:8.4f}  "
                f"{result['t_test_t']:7.4f}  "
                f"{result['p_value']:7.4f}"
            )

    def save_results(self, output_file: str) -> None:
        """Persist cross-validation results as a pickle file."""
        print(f"\nSaving results to {output_file}")
        save_data = {
            "results": self.results,
            "xgb_params": self.xgb_params,
            "embedding_file": self.embedding_file,
            "fold_dir": self.fold_dir,
        }

        with open(output_file, "wb") as f:
            pickle.dump(save_data, f)

        print("Results saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PTM SWING ten-fold XGBoost evaluation.")
    parser.add_argument("--embedding-file", required=True, help="Pickle file containing SWING embedding vectors.")
    parser.add_argument("--fold-dir", required=True, help="Directory containing fold_{i}_{train,test}.csv files.")
    parser.add_argument("--output-file", required=True, help="Output pickle path for fold results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    cv = PTMCrossValidator(args.embedding_file, args.fold_dir)
    cv.run_cross_validation()
    cv.save_results(args.output_file)

    print(f"\nTen-fold cross-validation complete: {args.output_file}")


if __name__ == "__main__":
    main()

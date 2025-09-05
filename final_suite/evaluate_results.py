#!/usr/bin/env python3
"""
Model Evaluation and Comparison Script
Evaluates and compares different anomaly detection methods
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionEvaluator:
    """Evaluate and compare anomaly detection results"""
    
    def __init__(self, results_csv_path, ground_truth_column=None):
        """
        Initialize evaluator
        
        Args:
            results_csv_path: Path to CSV with anomaly scores
            ground_truth_column: Column name containing ground truth labels (if available)
        """
        self.results_df = pd.read_csv(results_csv_path)
        self.ground_truth_column = ground_truth_column
        self.has_ground_truth = ground_truth_column is not None and ground_truth_column in self.results_df.columns
        
        # Identify score columns (those ending with '_score')
        self.score_columns = [col for col in self.results_df.columns if col.endswith('_score')]
        
        print(f"Loaded results with {len(self.results_df)} samples")
        print(f"Found {len(self.score_columns)} anomaly detection methods")
        print(f"Ground truth available: {self.has_ground_truth}")
        
    def calculate_threshold_metrics(self, scores, ground_truth, thresholds=None):
        """Calculate metrics across different thresholds"""
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        metrics = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'n_predictions': []
        }
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            
            if np.sum(predictions) == 0:  # No positive predictions
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = precision_score(ground_truth, predictions, zero_division=0)
                recall = recall_score(ground_truth, predictions, zero_division=0)
                f1 = f1_score(ground_truth, predictions, zero_division=0)
            
            accuracy = accuracy_score(ground_truth, predictions)
            
            metrics['threshold'].append(threshold)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['accuracy'].append(accuracy)
            metrics['n_predictions'].append(np.sum(predictions))
        
        return pd.DataFrame(metrics)
    
    def evaluate_with_ground_truth(self):
        """Evaluate methods when ground truth is available"""
        
        if not self.has_ground_truth:
            print("No ground truth available for evaluation")
            return None
        
        ground_truth = self.results_df[self.ground_truth_column].values
        
        print(f"\nEvaluating with ground truth:")
        print(f"Total anomalies in ground truth: {np.sum(ground_truth)} ({np.mean(ground_truth)*100:.2f}%)")
        
        evaluation_results = {}
        
        for method in self.score_columns:
            scores = self.results_df[method].values
            
            # Skip if all scores are the same (method failed)
            if np.std(scores) == 0:
                print(f"Skipping {method} (constant scores)")
                continue
            
            # Calculate AUC-ROC
            try:
                auc_roc = roc_auc_score(ground_truth, scores)
            except ValueError:
                auc_roc = np.nan
            
            # Calculate AUC-PR
            precision, recall, _ = precision_recall_curve(ground_truth, scores)
            auc_pr = np.trapz(recall, precision)
            
            # Find best threshold based on F1 score
            threshold_metrics = self.calculate_threshold_metrics(scores, ground_truth)
            best_f1_idx = threshold_metrics['f1'].idxmax()
            best_threshold = threshold_metrics.iloc[best_f1_idx]['threshold']
            best_f1 = threshold_metrics.iloc[best_f1_idx]['f1']
            
            # Calculate metrics at best threshold
            best_predictions = (scores > best_threshold).astype(int)
            best_precision = precision_score(ground_truth, best_predictions, zero_division=0)
            best_recall = recall_score(ground_truth, best_predictions, zero_division=0)
            best_accuracy = accuracy_score(ground_truth, best_predictions)
            
            evaluation_results[method] = {
                'AUC_ROC': auc_roc,
                'AUC_PR': auc_pr,
                'Best_Threshold': best_threshold,
                'Best_F1': best_f1,
                'Best_Precision': best_precision,
                'Best_Recall': best_recall,
                'Best_Accuracy': best_accuracy,
                'Threshold_Metrics': threshold_metrics
            }
        
        return evaluation_results
    
    def compare_methods_without_ground_truth(self):
        """Compare methods without ground truth using statistical measures"""
        
        print("\nComparing methods without ground truth:")
        
        comparison_results = {}
        
        for method in self.score_columns:
            scores = self.results_df[method].values
            
            # Skip if all scores are the same
            if np.std(scores) == 0:
                continue
            
            # Statistical properties
            stats = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'median_score': np.median(scores),
                'q95_score': np.percentile(scores, 95),
                'q99_score': np.percentile(scores, 99),
                'skewness': self._calculate_skewness(scores),
                'n_high_scores_5pct': np.sum(scores > np.percentile(scores, 95)),
                'n_high_scores_1pct': np.sum(scores > np.percentile(scores, 99)),
            }
            
            # Temporal clustering (are anomalies clustered in time?)
            high_score_indices = np.where(scores > np.percentile(scores, 95))[0]
            if len(high_score_indices) > 1:
                time_gaps = np.diff(high_score_indices)
                stats['avg_time_gap'] = np.mean(time_gaps)
                stats['clustering_ratio'] = np.sum(time_gaps == 1) / len(time_gaps)
            else:
                stats['avg_time_gap'] = np.nan
                stats['clustering_ratio'] = 0
            
            comparison_results[method] = stats
        
        return comparison_results
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def create_evaluation_plots(self, evaluation_results=None, save_plots=True):
        """Create comprehensive evaluation plots"""
        
        print("Creating evaluation plots...")
        
        # Determine number of methods and create subplots
        n_methods = len(self.score_columns)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        # Plot 1: Score distributions
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, method in enumerate(self.score_columns):
            ax = axes[i] if i < len(axes) else axes[-1]
            scores = self.results_df[method].values
            
            ax.hist(scores, bins=50, alpha=0.7, density=True)
            ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            ax.axvline(np.percentile(scores, 95), color='orange', linestyle='--', label='95th percentile')
            ax.set_title(f'{method} - Score Distribution')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('results/score_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Time series with anomaly scores
        fig, axes = plt.subplots(n_methods, 1, figsize=(15, 3*n_methods))
        if n_methods == 1:
            axes = [axes]
        
        time_index = range(len(self.results_df))
        
        for i, method in enumerate(self.score_columns):
            scores = self.results_df[method].values
            threshold = np.percentile(scores, 95)
            
            axes[i].plot(time_index, scores, alpha=0.8, linewidth=0.8)
            axes[i].axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'95th percentile: {threshold:.3f}')
            
            # Highlight high scores
            high_score_indices = np.where(scores > threshold)[0]
            if len(high_score_indices) > 0:
                axes[i].scatter(high_score_indices, scores[high_score_indices], 
                              c='red', s=20, alpha=0.8, label=f'High scores: {len(high_score_indices)}')
            
            axes[i].set_title(f'{method} - Anomaly Scores Over Time')
            axes[i].set_xlabel('Time Index')
            axes[i].set_ylabel('Anomaly Score')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('results/time_series_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Method comparison (if ground truth available)
        if evaluation_results and self.has_ground_truth:
            self._plot_roc_curves(evaluation_results, save_plots)
            self._plot_precision_recall_curves(evaluation_results, save_plots)
            self._plot_threshold_analysis(evaluation_results, save_plots)
    
    def _plot_roc_curves(self, evaluation_results, save_plots=True):
        """Plot ROC curves for all methods"""
        
        plt.figure(figsize=(10, 8))
        
        ground_truth = self.results_df[self.ground_truth_column].values
        
        for method, results in evaluation_results.items():
            if np.isnan(results['AUC_ROC']):
                continue
                
            scores = self.results_df[method].values
            fpr, tpr, _ = roc_curve(ground_truth, scores)
            
            plt.plot(fpr, tpr, linewidth=2, label=f"{method} (AUC: {results['AUC_ROC']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_precision_recall_curves(self, evaluation_results, save_plots=True):
        """Plot Precision-Recall curves for all methods"""
        
        plt.figure(figsize=(10, 8))
        
        ground_truth = self.results_df[self.ground_truth_column].values
        baseline = np.mean(ground_truth)
        
        for method, results in evaluation_results.items():
            scores = self.results_df[method].values
            precision, recall, _ = precision_recall_curve(ground_truth, scores)
            
            plt.plot(recall, precision, linewidth=2, 
                    label=f"{method} (AUC: {results['AUC_PR']:.3f})")
        
        plt.axhline(baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline: {baseline:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_threshold_analysis(self, evaluation_results, save_plots=True):
        """Plot threshold analysis for best performing method"""
        
        # Find best method based on F1 score
        best_method = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['Best_F1'])
        best_metrics = evaluation_results[best_method]['Threshold_Metrics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision, Recall, F1 vs Threshold
        ax1.plot(best_metrics['threshold'], best_metrics['precision'], 'b-', label='Precision', linewidth=2)
        ax1.plot(best_metrics['threshold'], best_metrics['recall'], 'r-', label='Recall', linewidth=2)
        ax1.plot(best_metrics['threshold'], best_metrics['f1'], 'g-', label='F1', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'{best_method} - Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Number of predictions vs Threshold
        ax2.plot(best_metrics['threshold'], best_metrics['n_predictions'], 'purple', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Number of Positive Predictions')
        ax2.set_title(f'{best_method} - Predictions vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix at best threshold
        ground_truth = self.results_df[self.ground_truth_column].values
        scores = self.results_df[best_method].values
        best_threshold = evaluation_results[best_method]['Best_Threshold']
        predictions = (scores > best_threshold).astype(int)
        
        cm = confusion_matrix(ground_truth, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax3, cmap='Blues')
        ax3.set_title(f'{best_method} - Confusion Matrix\n(Threshold: {best_threshold:.3f})')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Score distribution with threshold
        ax4.hist(scores, bins=50, alpha=0.7, density=True, label='All scores')
        ax4.axvline(best_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Best threshold: {best_threshold:.3f}')
        ax4.set_xlabel('Anomaly Score')
        ax4.set_ylabel('Density')
        ax4.set_title(f'{best_method} - Score Distribution with Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('results/threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, evaluation_results=None, comparison_results=None):
        """Generate comprehensive evaluation report"""
        
        print("Generating evaluation report...")
        
        report_path = 'results/evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("ANOMALY DETECTION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset Information:\n")
            f.write(f"- Total samples: {len(self.results_df)}\n")
            f.write(f"- Methods evaluated: {len(self.score_columns)}\n")
            f.write(f"- Ground truth available: {self.has_ground_truth}\n\n")
            
            if self.has_ground_truth and evaluation_results:
                f.write("SUPERVISED EVALUATION RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                # Create summary table
                summary_data = []
                for method, results in evaluation_results.items():
                    summary_data.append([
                        method,
                        f"{results['AUC_ROC']:.3f}",
                        f"{results['AUC_PR']:.3f}",
                        f"{results['Best_F1']:.3f}",
                        f"{results['Best_Precision']:.3f}",
                        f"{results['Best_Recall']:.3f}",
                        f"{results['Best_Threshold']:.3f}"
                    ])
                
                # Sort by F1 score
                summary_data.sort(key=lambda x: float(x[3]), reverse=True)
                
                f.write(f"{'Method':<25} {'AUC-ROC':<8} {'AUC-PR':<8} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Thresh':<7}\n")
                f.write("-" * 80 + "\n")
                
                for row in summary_data:
                    f.write(f"{row[0]:<25} {row[1]:<8} {row[2]:<8} {row[3]:<6} {row[4]:<6} {row[5]:<6} {row[6]:<7}\n")
                
                f.write(f"\nBest performing method: {summary_data[0][0]}\n\n")
            
            if comparison_results:
                f.write("UNSUPERVISED COMPARISON RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                for method, stats in comparison_results.items():
                    f.write(f"\n{method}:\n")
                    f.write(f"  Mean score: {stats['mean_score']:.4f}\n")
                    f.write(f"  Std score: {stats['std_score']:.4f}\n")
                    f.write(f"  95th percentile: {stats['q95_score']:.4f}\n")
                    f.write(f"  High scores (top 5%): {stats['n_high_scores_5pct']}\n")
                    f.write(f"  High scores (top 1%): {stats['n_high_scores_1pct']}\n")
                    f.write(f"  Clustering ratio: {stats['clustering_ratio']:.3f}\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            
            if self.has_ground_truth and evaluation_results:
                best_method = max(evaluation_results.keys(), 
                                key=lambda x: evaluation_results[x]['Best_F1'])
                f.write(f"1. Best overall method: {best_method}\n")
                f.write(f"2. Recommended threshold: {evaluation_results[best_method]['Best_Threshold']:.3f}\n")
                
                if evaluation_results[best_method]['Best_Recall'] < 0.7:
                    f.write("3. Consider lowering threshold to improve recall\n")
                if evaluation_results[best_method]['Best_Precision'] < 0.5:
                    f.write("3. Consider raising threshold to improve precision\n")
            else:
                f.write("1. No ground truth available - use ensemble methods\n")
                f.write("2. Focus on methods with consistent high scores\n")
                f.write("3. Investigate time periods with multiple method agreement\n")
        
        print(f"Evaluation report saved to: {report_path}")
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        
        print("Running complete evaluation...")
        
        # Evaluate with ground truth if available
        evaluation_results = None
        if self.has_ground_truth:
            evaluation_results = self.evaluate_with_ground_truth()
        
        # Compare methods without ground truth
        comparison_results = self.compare_methods_without_ground_truth()
        
        # Create plots
        self.create_evaluation_plots(evaluation_results)
        
        # Generate report
        self.generate_evaluation_report(evaluation_results, comparison_results)
        
        print("Evaluation completed successfully!")
        
        return evaluation_results, comparison_results


def main():
    """Main function for evaluation script"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection results")
    parser.add_argument('--results', '-r', default='results/detailed_anomaly_scores.csv',
                       help='Path to results CSV file')
    parser.add_argument('--ground-truth', '-g', 
                       help='Column name containing ground truth labels')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Please run the main detection script first.")
        return
    
    # Initialize evaluator
    evaluator = AnomalyDetectionEvaluator(args.results, args.ground_truth)
    
    # Run complete evaluation
    evaluator.run_complete_evaluation()
    
    print("\nEvaluation completed! Check the results/ directory for:")
    print("- evaluation_report.txt: Comprehensive evaluation report")
    print("- *.png files: Visualization plots")


if __name__ == "__main__":
    main()
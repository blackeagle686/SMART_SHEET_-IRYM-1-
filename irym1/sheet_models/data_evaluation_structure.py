# Abdo Kamal
"""
Advanced Model Evaluation and Reporting Pipeline

This module provides comprehensive evaluation capabilities for machine learning models
across Classification, Regression, and Clustering tasks. It generates detailed reports
with visualizations and supports multiple output formats.

Features:
- Comprehensive metrics for all ML task types
- Advanced visualizations and plots
- Cross-validation and hold-out testing
- Reproducible evaluation with detailed logging
- Multiple export formats (JSON, HTML, Markdown)
- SOLID principle-based architecture
- Extensible and modular design
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
import joblib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path

# Sklearn imports
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    # Regression metrics
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    # Clustering metrics
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class EvaluationConfig:
    """Configuration class for evaluation pipeline."""
    pipeline_type: str  # 'Classification', 'Regression', 'Clustering'
    model_path: str
    test_data_path: str
    target_column: str = 'target'
    output_dir: str = './evaluation_results'
    
    # Evaluation settings
    use_cross_validation: bool = True
    cv_folds: int = 5
    random_seed: int = 42
    
    # Metrics configuration
    classification_metrics: List[str] = None
    regression_metrics: List[str] = None
    clustering_metrics: List[str] = None
    
    # Visualization settings
    generate_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    
    # Report settings
    export_formats: List[str] = None
    include_feature_importance: bool = True
    
    def __post_init__(self):
        if self.classification_metrics is None:
            self.classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        if self.regression_metrics is None:
            self.regression_metrics = ['mae', 'mse', 'rmse', 'r2', 'mape']
        if self.clustering_metrics is None:
            self.clustering_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        if self.export_formats is None:
            self.export_formats = ['json', 'html']


class EvaluationResults:
    """Container for evaluation results."""
    
    def __init__(self):
        self.metrics = {}
        self.plots = {}
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.0.0'
        }
        self.feature_importance = {}
        self.confusion_matrix = None
        self.classification_report = None


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.results = EvaluationResults()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for evaluation pipeline."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.config.output_dir, 'evaluation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @abstractmethod
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pass
    
    @abstractmethod
    def generate_plots(self, model: Any, X: np.ndarray, y_true: np.ndarray) -> Dict[str, str]:
        """Generate evaluation plots."""
        pass
    
    def load_model_and_data(self) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Load model and test data."""
        self.logger.info("Loading model and test data...")
        
        # Load model
        model = joblib.load(self.config.model_path)
        self.logger.info(f"Model loaded from {self.config.model_path}")
        
        # Load test data
        if self.config.test_data_path.endswith('.csv'):
            data = pd.read_csv(self.config.test_data_path)
        elif self.config.test_data_path.endswith('.pkl'):
            data = pd.read_pickle(self.config.test_data_path)
        else:
            raise ValueError("Unsupported data format")
        
        # Split features and target
        if self.config.pipeline_type != 'Clustering':
            X = data.drop(columns=[self.config.target_column]).values
            y = data[self.config.target_column].values
        else:
            X = data.values
            y = None
        
        self.logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return model, X, y
    
    def evaluate(self) -> EvaluationResults:
        """Main evaluation method."""
        self.logger.info("Starting model evaluation...")
        
        try:
            # Load model and data
            model, X, y = self.load_model_and_data()
            
            # Set random seed for reproducibility
            np.random.seed(self.config.random_seed)
            
            # Compute metrics
            if self.config.use_cross_validation and y is not None:
                self.results.metrics = self._cross_validation_metrics(model, X, y)
            else:
                y_pred = model.predict(X)
                self.results.metrics = self.compute_metrics(y, y_pred)
            
            # Generate plots
            if self.config.generate_plots:
                self.results.plots = self.generate_plots(model, X, y)
            
            # Feature importance (if available)
            if self.config.include_feature_importance and hasattr(model, 'feature_importances_'):
                self.results.feature_importance = {
                    f'feature_{i}': importance 
                    for i, importance in enumerate(model.feature_importances_)
                }
            
            # Update metadata
            self.results.metadata.update({
                'model_type': type(model).__name__,
                'pipeline_type': self.config.pipeline_type,
                'data_shape': list(X.shape),
                'random_seed': self.config.random_seed,
                'cv_folds': self.config.cv_folds if self.config.use_cross_validation else None
            })
            
            self.logger.info("Evaluation completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def _cross_validation_metrics(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation evaluation."""
        self.logger.info(f"Performing {self.config.cv_folds}-fold cross-validation...")
        
        if self.config.pipeline_type == 'Classification':
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_seed)
            scoring = self.config.classification_metrics
        else:  # Regression
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                      random_state=self.config.random_seed)
            scoring = self.config.regression_metrics
        
        # Map metric names to sklearn scoring
        scoring_map = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_ovr_weighted',
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'r2': 'r2'
        }
        
        sklearn_scoring = [scoring_map.get(metric, metric) for metric in scoring if metric in scoring_map]
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=sklearn_scoring, 
                                  return_train_score=True, n_jobs=-1)
        
        metrics = {}
        for metric in scoring:
            if metric in scoring_map:
                score_key = f'test_{scoring_map[metric]}'
                if score_key in cv_results:
                    scores = cv_results[score_key]
                    # Handle negative scores (MAE, MSE)
                    if metric in ['mae', 'mse']:
                        scores = -scores
                    metrics[f'{metric}_mean'] = np.mean(scores)
                    metrics[f'{metric}_std'] = np.std(scores)
        
        return metrics


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification models."""
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        if 'accuracy' in self.config.classification_metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.config.classification_metrics:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        
        if 'recall' in self.config.classification_metrics:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        
        if 'f1' in self.config.classification_metrics:
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Store confusion matrix
        self.results.confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
        self.results.classification_report = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def generate_plots(self, model: Any, X: np.ndarray, y_true: np.ndarray) -> Dict[str, str]:
        """Generate classification plots."""
        plots = {}
        
        # Predictions
        y_pred = model.predict(X)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(self.config.output_dir, f'confusion_matrix.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        plots['confusion_matrix'] = plot_path
        
        # ROC Curve (for binary classification or multiclass with probabilities)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                
                if len(np.unique(y_true)) == 2:  # Binary classification
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    plt.grid(True)
                    
                    plot_path = os.path.join(self.config.output_dir, f'roc_curve.{self.config.plot_format}')
                    plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
                    plt.close()
                    plots['roc_curve'] = plot_path
                    
                    # Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                    avg_precision = average_precision_score(y_true, y_proba[:, 1])
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.2f})')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve')
                    plt.legend()
                    plt.grid(True)
                    
                    plot_path = os.path.join(self.config.output_dir, f'precision_recall_curve.{self.config.plot_format}')
                    plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
                    plt.close()
                    plots['precision_recall_curve'] = plot_path
        
        except Exception as e:
            self.logger.warning(f"Could not generate ROC/PR curves: {str(e)}")
        
        return plots


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression models."""
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        if 'mae' in self.config.regression_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'mse' in self.config.regression_metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        if 'rmse' in self.config.regression_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'r2' in self.config.regression_metrics:
            metrics['r2'] = r2_score(y_true, y_pred)
        
        if 'mape' in self.config.regression_metrics:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        return metrics
    
    def generate_plots(self, model: Any, X: np.ndarray, y_true: np.ndarray) -> Dict[str, str]:
        """Generate regression plots."""
        plots = {}
        
        y_pred = model.predict(X)
        
        # Residual Plot
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
        
        plot_path = os.path.join(self.config.output_dir, f'residual_plot.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        plots['residual_plot'] = plot_path
        
        # Prediction vs Actual
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.grid(True)
        
        plot_path = os.path.join(self.config.output_dir, f'prediction_vs_actual.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        plots['prediction_vs_actual'] = plot_path
        
        return plots


class ClusteringEvaluator(BaseEvaluator):
    """Evaluator for clustering models."""
    
    def compute_metrics(self, y_true: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering metrics."""
        metrics = {}
        
        if 'silhouette' in self.config.clustering_metrics:
            metrics['silhouette'] = silhouette_score(y_true, cluster_labels)
        
        if 'davies_bouldin' in self.config.clustering_metrics:
            metrics['davies_bouldin'] = davies_bouldin_score(y_true, cluster_labels)
        
        if 'calinski_harabasz' in self.config.clustering_metrics:
            metrics['calinski_harabasz'] = calinski_harabasz_score(y_true, cluster_labels)
        
        return metrics
    
    def generate_plots(self, model: Any, X: np.ndarray, y_true: np.ndarray = None) -> Dict[str, str]:
        """Generate clustering plots."""
        plots = {}
        
        cluster_labels = model.predict(X)
        
        # 2D visualization (using first two features or PCA)
        if X.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            
            if X.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                plt.title('Clustering Results (PCA Projection)')
            else:
                X_2d = X[:, :2]
                plt.title('Clustering Results')
            
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
            # Plot cluster centers if available
            if hasattr(model, 'cluster_centers_'):
                if X.shape[1] > 2:
                    centers_2d = pca.transform(model.cluster_centers_)
                else:
                    centers_2d = model.cluster_centers_[:, :2]
                
                plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                           marker='x', s=300, linewidths=3, color='red', label='Centroids')
                plt.legend()
            
            plot_path = os.path.join(self.config.output_dir, f'clustering_visualization.{self.config.plot_format}')
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            plots['clustering_visualization'] = plot_path
        
        return plots


class ReportGenerator:
    """Generate evaluation reports in different formats."""
    
    def __init__(self, config: EvaluationConfig, results: EvaluationResults):
        self.config = config
        self.results = results
    
    def generate_reports(self) -> Dict[str, str]:
        """Generate reports in specified formats."""
        report_paths = {}
        
        for fmt in self.config.export_formats:
            if fmt == 'json':
                report_paths['json'] = self._generate_json_report()
            elif fmt == 'html':
                report_paths['html'] = self._generate_html_report()
            elif fmt == 'markdown':
                report_paths['markdown'] = self._generate_markdown_report()
        
        return report_paths
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        report_data = {
            'metadata': self.results.metadata,
            'metrics': self.results.metrics,
            'feature_importance': self.results.feature_importance,
            'plots': self.results.plots,
            'confusion_matrix': self.results.confusion_matrix,
            'classification_report': self.results.classification_report
        }
        
        json_path = os.path.join(self.config.output_dir, 'evaluation_report.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return json_path
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ margin: 20px 0; }}
                .metric-item {{ display: inline-block; margin: 10px; padding: 10px; 
                              background-color: #e9ecef; border-radius: 5px; }}
                .plots {{ margin: 20px 0; }}
                .plot-item {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Evaluation Report</h1>
                <p><strong>Pipeline Type:</strong> {self.config.pipeline_type}</p>
                <p><strong>Model:</strong> {self.results.metadata.get('model_type', 'N/A')}</p>
                <p><strong>Evaluation Date:</strong> {self.results.metadata['timestamp']}</p>
                <p><strong>Data Shape:</strong> {self.results.metadata.get('data_shape', 'N/A')}</p>
            </div>
            
            <div class="metrics">
                <h2>Performance Metrics</h2>
                {self._format_metrics_html()}
            </div>
            
            {self._format_plots_html() if self.results.plots else ''}
            
            {self._format_feature_importance_html() if self.results.feature_importance else ''}
        </body>
        </html>
        """
        
        html_path = os.path.join(self.config.output_dir, 'evaluation_report.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _format_metrics_html(self) -> str:
        """Format metrics for HTML report."""
        html = ""
        for metric, value in self.results.metrics.items():
            if isinstance(value, float):
                html += f'<div class="metric-item"><strong>{metric}:</strong> {value:.4f}</div>'
            else:
                html += f'<div class="metric-item"><strong>{metric}:</strong> {value}</div>'
        return html
    
    def _format_plots_html(self) -> str:
        """Format plots for HTML report."""
        if not self.results.plots:
            return ""
        
        html = '<div class="plots"><h2>Visualizations</h2>'
        for plot_name, plot_path in self.results.plots.items():
            # Convert absolute path to relative for HTML
            rel_path = os.path.basename(plot_path)
            html += f'''
            <div class="plot-item">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{rel_path}" alt="{plot_name}" style="max-width: 100%; height: auto;">
            </div>
            '''
        html += '</div>'
        return html
    
    def _format_feature_importance_html(self) -> str:
        """Format feature importance for HTML report."""
        if not self.results.feature_importance:
            return ""
        
        html = '<div class="feature-importance"><h2>Feature Importance</h2><table>'
        html += '<tr><th>Feature</th><th>Importance</th></tr>'
        
        # Sort by importance
        sorted_features = sorted(self.results.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features:
            html += f'<tr><td>{feature}</td><td>{importance:.4f}</td></tr>'
        
        html += '</table></div>'
        return html
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        md_content = f"""# Model Evaluation Report

## Metadata
- **Pipeline Type**: {self.config.pipeline_type}
- **Model**: {self.results.metadata.get('model_type', 'N/A')}
- **Evaluation Date**: {self.results.metadata['timestamp']}
- **Data Shape**: {self.results.metadata.get('data_shape', 'N/A')}

## Performance Metrics
{self._format_metrics_markdown()}

{self._format_feature_importance_markdown() if self.results.feature_importance else ''}
"""
        
        md_path = os.path.join(self.config.output_dir, 'evaluation_report.md')
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        return md_path
    
    def _format_metrics_markdown(self) -> str:
        """Format metrics for Markdown report."""
        md = "| Metric | Value |\n|--------|-------|\n"
        for metric, value in self.results.metrics.items():
            if isinstance(value, float):
                md += f"| {metric} | {value:.4f} |\n"
            else:
                md += f"| {metric} | {value} |\n"
        return md
    
    def _format_feature_importance_markdown(self) -> str:
        """Format feature importance for Markdown report."""
        md = "\n## Feature Importance\n| Feature | Importance |\n|---------|------------|\n"
        
        # Sort by importance
        sorted_features = sorted(self.results.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features:
            md += f"| {feature} | {importance:.4f} |\n"
        
        return md


class EvaluationPipeline:
    """Main evaluation pipeline orchestrator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluator = self._get_evaluator()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger."""
        logger = logging.getLogger("EvaluationPipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _get_evaluator(self) -> BaseEvaluator:
        """Get appropriate evaluator based on pipeline type."""
        evaluator_map = {
            'Classification': ClassificationEvaluator,
            'Regression': RegressionEvaluator,
            'Clustering': ClusteringEvaluator
        }
        
        if self.config.pipeline_type not in evaluator_map:
            raise ValueError(f"Unsupported pipeline type: {self.config.pipeline_type}")
        
        return evaluator_map[self.config.pipeline_type](self.config)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        self.logger.info("Starting evaluation pipeline...")
        
        try:
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Run evaluation
            results = self.evaluator.evaluate()
            
            # Generate reports
            report_generator = ReportGenerator(self.config, results)
            report_paths = report_generator.generate_reports()
            
            self.logger.info("Evaluation pipeline completed successfully")
            
            return {
                'results': results,
                'report_paths': report_paths,
                'config': asdict(self.config)
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {str(e)}")
            raise


# Usage Example and Factory Functions
def create_evaluation_config(
    pipeline_type: str,
    model_path: str,
    test_data_path: str,
    output_dir: str = './evaluation_results',
    **kwargs
) -> EvaluationConfig:
    """Factory function to create evaluation configuration."""
    return EvaluationConfig(
        pipeline_type=pipeline_type,
        model_path=model_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        **kwargs
    )


def run_evaluation_pipeline(config: EvaluationConfig) -> Dict[str, Any]:
    """Convenience function to run evaluation pipeline."""
    pipeline = EvaluationPipeline(config)
    return pipeline.run()


def load_and_evaluate_model(
    pipeline_type: str,
    model_path: str,
    test_data_path: str,
    output_dir: str = './evaluation_results',
    **config_kwargs
) -> Dict[str, Any]:
    """
    High-level function to load and evaluate a model.
    
    Args:
        pipeline_type: Type of ML pipeline ('Classification', 'Regression', 'Clustering')
        model_path: Path to the trained model file
        test_data_path: Path to the test dataset
        output_dir: Directory to save evaluation results
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Dictionary containing evaluation results and report paths
    """
    config = create_evaluation_config(
        pipeline_type=pipeline_type,
        model_path=model_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        **config_kwargs
    )
    
    return run_evaluation_pipeline(config)


# Advanced Features
class ModelComparator:
    """Compare multiple models on the same dataset."""
    
    def __init__(self, test_data_path: str, output_dir: str = './model_comparison'):
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        self.results = {}
    
    def add_model(self, model_name: str, model_path: str, pipeline_type: str, **kwargs):
        """Add a model to comparison."""
        config = create_evaluation_config(
            pipeline_type=pipeline_type,
            model_path=model_path,
            test_data_path=self.test_data_path,
            output_dir=os.path.join(self.output_dir, model_name),
            **kwargs
        )
        
        pipeline = EvaluationPipeline(config)
        self.results[model_name] = pipeline.run()
    
    def generate_comparison_report(self) -> str:
        """Generate a comparison report for all models."""
        if not self.results:
            raise ValueError("No models added for comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['results'].metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        os.makedirs(self.output_dir, exist_ok=True)
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        # Generate comparison plot
        if len(comparison_df.columns) > 2:  # More than just Model column
            plt.figure(figsize=(12, 8))
            
            # Select numeric columns for plotting
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Create subplot for each metric
                n_metrics = len(numeric_cols)
                n_cols = min(3, n_metrics)
                n_rows = (n_metrics + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_metrics == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                for i, metric in enumerate(numeric_cols):
                    ax = axes[i] if n_metrics > 1 else axes[0]
                    comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, rot=45)
                    ax.set_title(f'{metric.title()} Comparison')
                    ax.set_ylabel(metric.title())
                
                # Hide empty subplots
                for i in range(n_metrics, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plot_path = os.path.join(self.output_dir, 'comparison_plot.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        return comparison_path


class EvaluationBenchmark:
    """Benchmark evaluation pipeline performance."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.benchmark_results = {}
    
    def run_benchmark(self, n_runs: int = 5) -> Dict[str, Any]:
        """Run evaluation pipeline multiple times and collect timing stats."""
        import time
        
        execution_times = []
        memory_usage = []
        
        for i in range(n_runs):
            start_time = time.time()
            
            # Run evaluation
            pipeline = EvaluationPipeline(self.config)
            results = pipeline.run()
            
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        self.benchmark_results = {
            'n_runs': n_runs,
            'execution_times': execution_times,
            'mean_time': np.mean(execution_times),
            'std_time': np.std(execution_times),
            'min_time': np.min(execution_times),
            'max_time': np.max(execution_times)
        }
        
        return self.benchmark_results


# Utility Functions
def validate_model_file(model_path: str) -> bool:
    """Validate if model file exists and is loadable."""
    try:
        if not os.path.exists(model_path):
            return False
        
        # Try to load the model
        joblib.load(model_path)
        return True
    except Exception:
        return False


def validate_data_file(data_path: str, target_column: str = 'target') -> Dict[str, Any]:
    """Validate test data file and return basic info."""
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.pkl'):
            data = pd.read_pickle(data_path)
        else:
            raise ValueError("Unsupported file format")
        
        info = {
            'valid': True,
            'shape': data.shape,
            'columns': list(data.columns),
            'has_target': target_column in data.columns,
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        return info
    
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def create_sample_config() -> EvaluationConfig:
    """Create a sample configuration for testing."""
    return EvaluationConfig(
        pipeline_type='Classification',
        model_path='./models/trained_model.pkl',
        test_data_path='./data/test_data.csv',
        output_dir='./evaluation_results',
        use_cross_validation=True,
        cv_folds=5,
        generate_plots=True,
        export_formats=['json', 'html', 'markdown']
    )



#use case examples

"""
if __name__ == "__main__":
    
    # Example 1: Basic evaluation
    print("Example 1: Basic Model Evaluation")
    
    # Create configuration
    config = EvaluationConfig(
        pipeline_type='Classification',
        model_path='./models/classifier_model.pkl',
        test_data_path='./data/test_data.csv',
        output_dir='./evaluation_results/classifier',
        use_cross_validation=True,
        cv_folds=5,
        generate_plots=True,
        export_formats=['json', 'html']
    )
    
    # Run evaluation
    try:
        results = run_evaluation_pipeline(config)
        print("✅ Classification evaluation completed successfully!")
        print(f"📊 Metrics: {results['results'].metrics}")
        print(f"📁 Reports saved to: {results['report_paths']}")
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Model comparison
    print("Example 2: Model Comparison")
    
    try:
        comparator = ModelComparator('./data/test_data.csv', './model_comparison')
        
        # Add multiple models
        comparator.add_model(
            'RandomForest', 
            './models/rf_model.pkl', 
            'Classification'
        )
        comparator.add_model(
            'SVM', 
            './models/svm_model.pkl', 
            'Classification'
        )
        
        # Generate comparison report
        comparison_path = comparator.generate_comparison_report()
        print(f"✅ Model comparison completed!")
        print(f"📊 Comparison report: {comparison_path}")
    except Exception as e:
        print(f"❌ Model comparison failed: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Convenience function
    print("Example 3: Using Convenience Function")
    
    try:
        results = load_and_evaluate_model(
            pipeline_type='Regression',
            model_path='./models/regression_model.pkl',
            test_data_path='./data/regression_test.csv',
            output_dir='./evaluation_results/regression',
            use_cross_validation=False,
            generate_plots=True
        )
        print("✅ Regression evaluation completed!")
        print(f"📊 R² Score: {results['results'].metrics.get('r2', 'N/A')}")
    except Exception as e:
        print(f"❌ Regression evaluation failed: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Clustering evaluation
    print("Example 4: Clustering Evaluation")
    
    try:
        clustering_config = EvaluationConfig(
            pipeline_type='Clustering',
            model_path='./models/kmeans_model.pkl',
            test_data_path='./data/clustering_data.csv',
            output_dir='./evaluation_results/clustering',
            clustering_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
            generate_plots=True
        )
        
        results = run_evaluation_pipeline(clustering_config)
        print("✅ Clustering evaluation completed!")
        print(f"📊 Silhouette Score: {results['results'].metrics.get('silhouette', 'N/A')}")
    except Exception as e:
        print(f"❌ Clustering evaluation failed: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    print("🎉 Evaluation pipeline examples completed!")
    print("\nKey Features:")
    print("• Comprehensive metrics for all ML task types")
    print("• Advanced visualizations and plots")
    print("• Cross-validation and hold-out testing")
    print("• Multiple export formats (JSON, HTML, Markdown)")
    print("• Model comparison capabilities")
    print("• Reproducible evaluation with detailed logging")
    print("• SOLID principle-based architecture")
    print("• Extensible and modular design")

"""
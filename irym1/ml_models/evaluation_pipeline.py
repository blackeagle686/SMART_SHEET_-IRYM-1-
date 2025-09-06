import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    roc_curve, auc, precision_recall_curve
)

class EvaluationPipeline:
    """
    Evaluation Pipeline for trained ML models
    Generates comprehensive evaluation reports and visualizations
    """
    
    def __init__(self, model_path: str, test_data_path: str, pipeline_type: str, output_dir: str):
        """
        Initialize Evaluation Pipeline
        Takes: model_path, test_data_path, pipeline_type, output_dir
        Returns: EvaluationPipeline instance
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.pipeline_type = pipeline_type
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger for evaluation pipeline
        Takes: None
        Returns: Logger instance
        """
        logger = logging.getLogger(f"EvaluationPipeline_{self.pipeline_type}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.output_dir, 'evaluation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def load_model_and_data(self) -> tuple:
        """
        Load trained model and test data
        Takes: None
        Returns: Tuple of (model, X_test, y_test)
        """
        try:
            self.logger.info("Loading model and test data")
            
            # Load model
            model = joblib.load(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
            
            # Load test data
            if self.test_data_path.endswith('.csv'):
                df = pd.read_csv(self.test_data_path)
            elif self.test_data_path.endswith('.pkl'):
                df = pd.read_pickle(self.test_data_path)
            elif self.test_data_path.endswith('.npy'):
                data = np.load(self.test_data_path, allow_pickle=True)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {self.test_data_path}")
            
            # Split features and target
            if self.pipeline_type != 'Clustering':
                # Assume target column is named 'target' or is the last column
                if 'target' in df.columns:
                    X_test = df.drop(columns=['target'])
                    y_test = df['target']
                else:
                    X_test = df.iloc[:, :-1]
                    y_test = df.iloc[:, -1]
            else:
                X_test = df
                y_test = None
            
            self.logger.info(f"Data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            return model, X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Error loading model and data: {str(e)}")
            raise
    
    def evaluate_classification(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate classification model
        Takes: trained model, test features, test targets
        Returns: Dictionary with classification metrics and plots
        """
        try:
            self.logger.info("Evaluating classification model")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Generate plots
            plots = self._generate_classification_plots(y_test, y_pred, model, X_test)
            
            return {
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'plots': plots
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating classification model: {str(e)}")
            raise
    
    def evaluate_regression(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate regression model
        Takes: trained model, test features, test targets
        Returns: Dictionary with regression metrics and plots
        """
        try:
            self.logger.info("Evaluating regression model")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Generate plots
            plots = self._generate_regression_plots(y_test, y_pred)
            
            return {
                'metrics': metrics,
                'plots': plots
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating regression model: {str(e)}")
            raise
    
    def evaluate_clustering(self, model: Any, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate clustering model
        Takes: trained model, test features
        Returns: Dictionary with clustering metrics and plots
        """
        try:
            self.logger.info("Evaluating clustering model")
            
            # Get cluster labels
            labels = model.predict(X_test)
            
            # Calculate metrics (only if more than one cluster)
            metrics = {}
            if len(set(labels)) > 1:
                metrics['silhouette'] = silhouette_score(X_test, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(X_test, labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(X_test, labels)
            
            # Generate plots
            plots = self._generate_clustering_plots(X_test, labels, model)
            
            return {
                'metrics': metrics,
                'plots': plots
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating clustering model: {str(e)}")
            raise
    
    def _generate_classification_plots(self, y_test: pd.Series, y_pred: np.ndarray, 
                                     model: Any, X_test: pd.DataFrame) -> Dict[str, str]:
        """
        Generate classification plots
        Takes: true labels, predicted labels, model, test features
        Returns: Dictionary with plot file paths
        """
        plots = {}
        
        try:
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plot_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['confusion_matrix'] = plot_path
            
            # ROC Curve (for binary classification)
            if len(set(y_test)) == 2 and hasattr(model, 'predict_proba'):
                plt.figure(figsize=(8, 6))
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join(self.output_dir, 'roc_curve.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['roc_curve'] = plot_path
            
        except Exception as e:
            self.logger.warning(f"Error generating classification plots: {str(e)}")
        
        return plots
    
    def _generate_regression_plots(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, str]:
        """
        Generate regression plots
        Takes: true values, predicted values
        Returns: Dictionary with plot file paths
        """
        plots = {}
        
        try:
            # Residual Plot
            residuals = y_test - y_pred
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True)
            
            plot_path = os.path.join(self.output_dir, 'residual_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['residual_plot'] = plot_path
            
            # Prediction vs Actual
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.grid(True)
            
            plot_path = os.path.join(self.output_dir, 'prediction_vs_actual.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['prediction_vs_actual'] = plot_path
            
        except Exception as e:
            self.logger.warning(f"Error generating regression plots: {str(e)}")
        
        return plots
    
    def _generate_clustering_plots(self, X_test: pd.DataFrame, labels: np.ndarray, model: Any) -> Dict[str, str]:
        """
        Generate clustering plots
        Takes: test features, cluster labels, model
        Returns: Dictionary with plot file paths
        """
        plots = {}
        
        try:
            # 2D visualization (using first two features or PCA)
            plt.figure(figsize=(10, 8))
            
            if X_test.shape[1] >= 2:
                if X_test.shape[1] > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_test)
                    plt.title('Clustering Results (PCA Projection)')
                else:
                    X_2d = X_test.iloc[:, :2].values
                    plt.title('Clustering Results')
                
                scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
                plt.colorbar(scatter)
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                
                # Plot cluster centers if available
                if hasattr(model, 'cluster_centers_'):
                    if X_test.shape[1] > 2:
                        centers_2d = pca.transform(model.cluster_centers_)
                    else:
                        centers_2d = model.cluster_centers_[:, :2]
                    
                    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                               marker='x', s=300, linewidths=3, color='red', label='Centroids')
                    plt.legend()
                
                plot_path = os.path.join(self.output_dir, 'clustering_visualization.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['clustering_visualization'] = plot_path
            
        except Exception as e:
            self.logger.warning(f"Error generating clustering plots: {str(e)}")
        
        return plots
    
    def get_feature_importance(self, model: Any, feature_names: list) -> Dict[str, float]:
        """
        Extract feature importance if available
        Takes: trained model, list of feature names
        Returns: Dictionary with feature importance scores
        """
        try:
            feature_importance = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                    feature_importance[feature_name] = float(importance)
            
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_
                if len(coef.shape) == 1:  # Binary classification or regression
                    for i, coef_val in enumerate(coef):
                        feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                        feature_importance[feature_name] = float(abs(coef_val))
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {str(e)}")
            return {}
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline
        Takes: None
        Returns: Dictionary with evaluation results
        """
        try:
            self.logger.info("Starting evaluation pipeline")
            start_time = datetime.now()
            
            # Load model and data
            model, X_test, y_test = self.load_model_and_data()
            
            # Run evaluation based on pipeline type
            if self.pipeline_type == 'Classification':
                results = self.evaluate_classification(model, X_test, y_test)
            elif self.pipeline_type == 'Regression':
                results = self.evaluate_regression(model, X_test, y_test)
            elif self.pipeline_type == 'Clustering':
                results = self.evaluate_clustering(model, X_test)
            else:
                raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
            
            # Get feature importance
            feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else []
            feature_importance = self.get_feature_importance(model, feature_names)
            results['feature_importance'] = feature_importance
            
            # Calculate evaluation time
            end_time = datetime.now()
            evaluation_time = (end_time - start_time).total_seconds()
            results['evaluation_time'] = evaluation_time
            
            # Save results
            results_path = os.path.join(self.output_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation pipeline completed in {evaluation_time:.2f} seconds")
            
            return {
                'success': True,
                'metrics': results['metrics'],
                'plots': results.get('plots', {}),
                'confusion_matrix': results.get('confusion_matrix'),
                'classification_report': results.get('classification_report'),
                'feature_importance': feature_importance,
                'evaluation_time': evaluation_time
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
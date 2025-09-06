import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score

# Import your existing training models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans



class TrainingPipeline:
    """
    Training Pipeline for ML models
    Integrates with Django models and handles training workflow
    """
    
    MODELS = {
        "Logistic Regression": LogisticRegression,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge,
        "Lasso Regression": Lasso,
        "Decision Tree Classifier": DecisionTreeClassifier,
        "Decision Tree Regressor": DecisionTreeRegressor,
        "Random Forest Classifier": RandomForestClassifier,
        "Random Forest Regressor": RandomForestRegressor,
        "Gradient Boosting Classifier": GradientBoostingClassifier,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "Support Vector Classifier": SVC,
        "Support Vector Regressor": SVR,
        "Gaussian Naive Bayes": GaussianNB,
        "KNN Classifier": KNeighborsClassifier,
        "KNN Regressor": KNeighborsRegressor,
        "KMeans Clustering": KMeans,
    }
    
    def __init__(self, project_id: int, dataset_path: str, config: Any, output_dir: str):
        """
        Initialize Training Pipeline
        Takes: project_id, dataset_path, ML configuration, output directory
        Returns: TrainingPipeline instance
        """
        self.project_id = project_id
        self.dataset_path = dataset_path
        self.config = config
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger for training pipeline
        Takes: None  
        Returns: Logger instance
        """
        logger = logging.getLogger(f"TrainingPipeline_{self.project_id}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def load_data(self) -> tuple:
        """
        Load dataset from file
        Takes: None
        Returns: Tuple of (X, y) where y is None for clustering
        """
        try:
            self.logger.info(f"Loading data from {self.dataset_path}")
            
            if self.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith('.pkl'):
                df = pd.read_pickle(self.dataset_path)
            elif self.dataset_path.endswith('.npy'):
                data = np.load(self.dataset_path, allow_pickle=True)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {self.dataset_path}")
            
            self.logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Split features and target
            if self.config.project.pipeline_type != 'Clustering':
                if not hasattr(self.config, 'project') or not hasattr(self.config.project, 'datasetinfo'):
                    raise ValueError("Dataset info not found")
                
                target_column = self.config.project.datasetinfo.target_column
                if target_column and target_column in df.columns:
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                else:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")
            else:
                X = df
                y = None
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, X: pd.DataFrame) -> tuple:
        """
        Preprocess the data (scaling, encoding)
        Takes: Feature DataFrame X
        Returns: Tuple of (preprocessed_X, scaler)
        """
        try:
            self.logger.info("Preprocessing data")
            
            # Handle categorical variables (simple label encoding for now)
            X_processed = X.copy()
            categorical_columns = X_processed.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                X_processed[col] = pd.Categorical(X_processed[col]).codes
            
            # Scale numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            # Save scaler
            scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
            joblib.dump(scaler, scaler_path)
            
            self.logger.info("Data preprocessing completed")
            return X_scaled, scaler
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def create_model(self) -> Any:
        """
        Create ML model based on configuration
        Takes: None
        Returns: Initialized ML model
        """
        try:
            algorithm = self.config.algorithm
            hyperparameters = self.config.hyperparameters
            
            if algorithm not in self.MODELS:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Set random seed
            if 'random_state' not in hyperparameters:
                hyperparameters['random_state'] = self.config.random_seed
            
            model = self.MODELS[algorithm](**hyperparameters)
            
            self.logger.info(f"Created model: {algorithm} with parameters: {hyperparameters}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise
    
    def train_model(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> Any:
        """
        Train the ML model
        Takes: model instance, features X, target y (optional for clustering)
        Returns: Trained model
        """
        try:
            self.logger.info("Starting model training")
            start_time = datetime.now()
            
            if self.config.project.pipeline_type == 'Clustering':
                model.fit(X)
            else:
                if y is None:
                    raise ValueError("Target variable required for supervised learning")
                model.fit(X, y)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Model training completed in {training_time:.2f} seconds")
            return model, training_time
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate trained model
        Takes: trained model, test features X, test target y
        Returns: Dictionary of evaluation metrics
        """
        try:
            self.logger.info("Evaluating model")
            metrics = {}
            
            pipeline_type = self.config.project.pipeline_type
            
            if pipeline_type == 'Classification':
                y_pred = model.predict(X)
                metrics['accuracy'] = accuracy_score(y, y_pred)
                
                # Add precision, recall, f1 if possible
                from sklearn.metrics import precision_score, recall_score, f1_score
                try:
                    metrics['precision'] = precision_score(y, y_pred, average='weighted')
                    metrics['recall'] = recall_score(y, y_pred, average='weighted')
                    metrics['f1'] = f1_score(y, y_pred, average='weighted')
                except:
                    pass
                    
            elif pipeline_type == 'Regression':
                y_pred = model.predict(X)
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                
                from sklearn.metrics import r2_score, mean_absolute_error
                metrics['r2'] = r2_score(y, y_pred)
                metrics['mae'] = mean_absolute_error(y, y_pred)
                
            elif pipeline_type == 'Clustering':
                labels = model.predict(X)
                if len(set(labels)) > 1:  # More than one cluster
                    metrics['silhouette'] = silhouette_score(X, labels)
                    
                    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
                    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model: Any) -> str:
        """
        Save trained model to file
        Takes: trained model
        Returns: Path to saved model file
        """
        try:
            model_path = os.path.join(self.output_dir, 'model.pkl')
            joblib.dump(model, model_path)
            
            self.logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def save_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Save evaluation metrics to JSON file
        Takes: metrics dictionary
        Returns: Path to saved metrics file
        """
        try:
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Metrics saved to {metrics_path}")
            return metrics_path
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        Takes: None
        Returns: Dictionary with training results
        """
        try:
            self.logger.info("Starting training pipeline")
            
            # Load data
            X, y = self.load_data()
            
            # Split data if not clustering
            if self.config.project.pipeline_type != 'Clustering':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=self.config.validation_split,
                    random_state=self.config.random_seed,
                    stratify=y if self.config.project.pipeline_type == 'Classification' else None
                )
            else:
                # For clustering, use all data for training
                X_train, y_train = X, None
                X_test, y_test = X, None
            
            # Preprocess data
            X_train_processed, scaler = self.preprocess_data(X_train)
            
            if self.config.project.pipeline_type != 'Clustering':
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_train_processed
            
            # Create and train model
            model = self.create_model()
            trained_model, training_time = self.train_model(model, X_train_processed, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test_processed, y_test)
            
            # Save model and results
            model_path = self.save_model(trained_model)
            metrics_path = self.save_metrics(metrics)
            
            # Save training configuration
            config_data = {
                'project_id': self.project_id,
                'algorithm': self.config.algorithm,
                'hyperparameters': self.config.hyperparameters,
                'pipeline_type': self.config.project.pipeline_type,
                'validation_split': self.config.validation_split,
                'random_seed': self.config.random_seed,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat()
            }
            
            config_path = os.path.join(self.output_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info("Training pipeline completed successfully")
            
            return {
                'success': True,
                'model_path': model_path,
                'metrics': metrics,
                'training_time': training_time,
                'config_path': config_path
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
class MLengine2:
    """
    Main ML Engine class to manage multiple training pipelines
    Uses X and y directly for training multiple models
    """
    
    def __init__(self, project_id: int, dataset: pd.DataFrame, configs: List[Any], output_base_dir: str):
        """
        Initialize MLengine2
        Takes: project_id, dataset (DataFrame containing X and y), list of configurations, output base directory
        Returns: MLengine2 instance
        """
        self.project_id = project_id
        self.dataset = dataset
        self.configs = configs
        self.output_base_dir = output_base_dir
        self.logger = self._setup_logger()
        
        # Create base output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Extract X and y from dataset
        self.X, self.y = self._prepare_data()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger for MLengine2
        Takes: None
        Returns: Logger instance
        """
        logger = logging.getLogger(f"MLengine2_{self.project_id}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.output_base_dir, 'mlengine2.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _prepare_data(self) -> tuple:
        """
        Prepare X and y from the provided dataset
        Takes: None
        Returns: Tuple of (X, y) where y is None for clustering
        """
        try:
            self.logger.info("Preparing data from provided dataset")
            
            if not isinstance(self.dataset, pd.DataFrame):
                raise ValueError("Dataset must be a pandas DataFrame")
            
            # Check pipeline type in the first config (assuming all configs share the same pipeline type)
            pipeline_type = self.configs[0].project.pipeline_type
            
            if pipeline_type != 'Clustering':
                if not hasattr(self.configs[0], 'project') or not hasattr(self.configs[0].project, 'datasetinfo'):
                    raise ValueError("Dataset info not found in configuration")
                
                target_column = self.configs[0].project.datasetinfo.target_column
                if target_column and target_column in self.dataset.columns:
                    X = self.dataset.drop(columns=[target_column])
                    y = self.dataset[target_column]
                else:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")
            else:
                X = self.dataset
                y = None
            
            self.logger.info(f"Data prepared: {X.shape[0]} rows, {X.shape[1]} columns")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def run_pipelines(self) -> List[Dict[str, Any]]:
        """
        Run training pipelines for all configurations
        Takes: None
        Returns: List of results from each pipeline
        """
        try:
            self.logger.info("Starting MLengine2 to run multiple pipelines")
            results = []
            
            for idx, config in enumerate(self.configs):
                try:
                    # Create unique output directory for this pipeline
                    output_dir = os.path.join(self.output_base_dir, f"pipeline_{idx}_{config.algorithm}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    self.logger.info(f"Running pipeline {idx+1}/{len(self.configs)} with algorithm: {config.algorithm}")
                    
                    # Initialize pipeline with dummy dataset_path (since we're passing X, y directly)
                    pipeline = TrainingPipeline(
                        project_id=self.project_id,
                        dataset_path="direct_data",  # Dummy path, not used
                        config=config,
                        output_dir=output_dir
                    )
                    
                    # Override pipeline's load_data to use self.X and self.y directly
                    def custom_load_data():
                        return self.X, self.y
                    
                    pipeline.load_data = custom_load_data
                    
                    # Run the pipeline
                    result = pipeline.run()
                    
                    # Add additional info to result
                    result['pipeline_id'] = idx
                    result['algorithm'] = config.algorithm
                    results.append(result)
                    
                    self.logger.info(f"Pipeline {idx+1} completed: {config.algorithm}")
                    
                except Exception as e:
                    self.logger.error(f"Pipeline {idx+1} failed for {config.algorithm}: {str(e)}")
                    results.append({
                        'success': False,
                        'pipeline_id': idx,
                        'algorithm': config.algorithm,
                        'error': str(e)
                    })
            
            # Save summary of results
            summary_path = os.path.join(self.output_base_dir, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"All pipelines completed. Summary saved to {summary_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"MLengine2 failed: {str(e)}")
            raise
    
    def get_best_model(self, results: List[Dict[str, Any]], metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """
        Select the best model based on a specified metric
        Takes: List of pipeline results, metric to optimize
        Returns: Best pipeline result or None if no successful results
        """
        try:
            self.logger.info(f"Selecting best model based on {metric}")
            
            # Filter successful results
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                self.logger.warning("No successful pipeline results found")
                return None
            
            # Find the best result based on the specified metric
            best_result = None
            best_score = float('-inf') if metric != 'mse' else float('inf')
            
            for result in successful_results:
                if 'metrics' in result and metric in result['metrics']:
                    score = result['metrics'][metric]
                    if metric in ['mse', 'rmse', 'mae', 'davies_bouldin']:  # Lower is better
                        if score < best_score:
                            best_score = score
                            best_result = result
                    else:  # Higher is better (e.g., accuracy, r2, silhouette)
                        if score > best_score:
                            best_score = score
                            best_result = result
            
            if best_result:
                self.logger.info(f"Best model: {best_result['algorithm']} with {metric} = {best_score}")
            else:
                self.logger.warning(f"No results found with metric {metric}")
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Error selecting best model: {str(e)}")
            raise
    
    def predict(self, model_path: str, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a saved model
        Takes: Path to saved model, new feature DataFrame X_new
        Returns: Array of predictions
        """
        try:
            self.logger.info(f"Making predictions with model: {model_path}")
            
            # Load model
            model = joblib.load(model_path)
            
            # Load scaler (assumed to be in the same directory as the model)
            scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                X_new_processed = scaler.transform(X_new)
            else:
                self.logger.warning("Scaler not found, using raw data for prediction")
                X_new_processed = X_new
            
            # Handle categorical variables (consistent with preprocessing)
            X_new_processed = pd.DataFrame(X_new_processed, columns=X_new.columns)
            categorical_columns = X_new_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X_new_processed[col] = pd.Categorical(X_new_processed[col]).codes
            
            # Make predictions
            predictions = model.predict(X_new_processed)
            
            self.logger.info(f"Predictions generated for {X_new.shape[0]} samples")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
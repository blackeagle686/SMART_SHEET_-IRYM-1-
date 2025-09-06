from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Configuration Generator", version="1.0.0")

class DatasetInfo(BaseModel):
    """
    Dataset information model
    """
    num_rows: int = Field(..., description="Number of rows in dataset")
    num_columns: int = Field(..., description="Number of columns in dataset")
    column_names: List[str] = Field(..., description="List of column names")
    column_types: Dict[str, str] = Field(..., description="Column data types")
    target_column: Optional[str] = Field(None, description="Target column name")
    has_missing_values: bool = Field(False, description="Whether dataset has missing values")
    pipeline_type: str = Field(..., description="Type of ML pipeline")

class ConfigRequest(BaseModel):
    """
    Configuration request model
    """
    dataset_info: DatasetInfo
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    project_name: str = Field(..., description="Name of the project")
    project_description: Optional[str] = Field(None, description="Project description")

class MLConfig(BaseModel):
    """
    Generated ML configuration model
    """
    algorithm: str = Field(..., description="Recommended algorithm")
    hyperparameters: Dict[str, Any] = Field(..., description="Recommended hyperparameters")
    reasoning: str = Field(..., description="Reasoning for algorithm choice")
    validation_split: float = Field(0.2, description="Validation split ratio")
    cross_validation_folds: int = Field(5, description="Number of CV folds")
    use_cross_validation: bool = Field(True, description="Whether to use CV")
    random_seed: int = Field(42, description="Random seed for reproducibility")

class EvaluationConfig(BaseModel):
    """
    Generated evaluation configuration model
    """
    metrics: List[str] = Field(..., description="Recommended evaluation metrics")
    generate_plots: bool = Field(True, description="Whether to generate plots")
    include_feature_importance: bool = Field(True, description="Include feature importance")
    export_formats: List[str] = Field(["json", "html"], description="Export formats")
    use_cross_validation: bool = Field(True, description="Use CV for evaluation")
    cv_folds: int = Field(5, description="Number of CV folds")

class ConfigResponse(BaseModel):
    """
    Complete configuration response model
    """
    ml_config: MLConfig
    evaluation_config: EvaluationConfig
    preprocessing_suggestions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ConfigGenerator:
    """
    ML Configuration Generator using rule-based recommendations
    """
    
    # Algorithm recommendations based on dataset characteristics
    CLASSIFICATION_ALGORITHMS = {
        "small_dataset": {
            "algorithm": "Logistic Regression",
            "hyperparameters": {"max_iter": 1000, "random_state": 42},
            "reasoning": "Logistic Regression works well with small datasets and provides interpretable results"
        },
        "medium_dataset": {
            "algorithm": "Random Forest Classifier", 
            "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "reasoning": "Random Forest handles medium-sized datasets well and provides feature importance"
        },
        "large_dataset": {
            "algorithm": "Gradient Boosting Classifier",
            "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
            "reasoning": "Gradient Boosting often performs well on large datasets with complex patterns"
        },
        "high_dimensional": {
            "algorithm": "Support Vector Classifier",
            "hyperparameters": {"C": 1.0, "kernel": "rbf", "random_state": 42},
            "reasoning": "SVM works well with high-dimensional data"
        }
    }
    
    REGRESSION_ALGORITHMS = {
        "small_dataset": {
            "algorithm": "Linear Regression",
            "hyperparameters": {},
            "reasoning": "Linear Regression is suitable for small datasets and provides interpretability"
        },
        "medium_dataset": {
            "algorithm": "Random Forest Regressor",
            "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "reasoning": "Random Forest Regressor handles medium-sized datasets well with good performance"
        },
        "large_dataset": {
            "algorithm": "Gradient Boosting Regressor",
            "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
            "reasoning": "Gradient Boosting often achieves excellent performance on large regression datasets"
        },
        "regularization_needed": {
            "algorithm": "Ridge Regression",
            "hyperparameters": {"alpha": 1.0, "random_state": 42},
            "reasoning": "Ridge Regression helps prevent overfitting when regularization is needed"
        }
    }
    
    CLUSTERING_ALGORITHMS = {
        "default": {
            "algorithm": "KMeans Clustering",
            "hyperparameters": {"n_clusters": 3, "random_state": 42, "n_init": 10},
            "reasoning": "K-Means is a good starting point for clustering analysis"
        }
    }
    
    def __init__(self):
        """
        Initialize configuration generator
        """
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_dataset_characteristics(self, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """
        Analyze dataset characteristics to determine best algorithm approach
        Takes: DatasetInfo object
        Returns: Dictionary with dataset characteristics
        """
        try:
            characteristics = {}
            
            # Dataset size categorization
            if dataset_info.num_rows < 1000:
                characteristics["size_category"] = "small_dataset"
            elif dataset_info.num_rows < 10000:
                characteristics["size_category"] = "medium_dataset"
            else:
                characteristics["size_category"] = "large_dataset"
            
            # Dimensionality analysis
            if dataset_info.num_columns > 50:
                characteristics["dimensionality"] = "high_dimensional"
            else:
                characteristics["dimensionality"] = "normal_dimensional"
            
            # Data type analysis
            numeric_columns = sum(1 for dtype in dataset_info.column_types.values() 
                                if 'int' in dtype or 'float' in dtype)
            categorical_columns = dataset_info.num_columns - numeric_columns
            
            characteristics["numeric_ratio"] = numeric_columns / dataset_info.num_columns
            characteristics["categorical_ratio"] = categorical_columns / dataset_info.num_columns
            
            # Missing data analysis
            characteristics["has_missing_data"] = dataset_info.has_missing_values
            
            self.logger.info(f"Dataset characteristics analyzed: {characteristics}")
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset: {str(e)}")
            raise
    
    def generate_ml_config(self, dataset_info: DatasetInfo, characteristics: Dict[str, Any]) -> MLConfig:
        """
        Generate ML training configuration
        Takes: DatasetInfo and dataset characteristics
        Returns: MLConfig with recommended settings
        """
        try:
            pipeline_type = dataset_info.pipeline_type
            
            # Select algorithm based on pipeline type and characteristics
            if pipeline_type == "Classification":
                algorithm_map = self.CLASSIFICATION_ALGORITHMS
            elif pipeline_type == "Regression":
                algorithm_map = self.REGRESSION_ALGORITHMS
            elif pipeline_type == "Clustering":
                algorithm_map = self.CLUSTERING_ALGORITHMS
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
            # Choose best algorithm based on characteristics
            if pipeline_type == "Clustering":
                config = algorithm_map["default"]
                # Adjust number of clusters based on dataset size
                if dataset_info.num_rows > 1000:
                    config["hyperparameters"]["n_clusters"] = min(10, max(3, dataset_info.num_rows // 100))
            else:
                # Priority order for algorithm selection
                if characteristics["dimensionality"] == "high_dimensional":
                    config = algorithm_map.get("high_dimensional", algorithm_map["medium_dataset"])
                else:
                    config = algorithm_map[characteristics["size_category"]]
                
                # Special case for regression with many features
                if (pipeline_type == "Regression" and 
                    dataset_info.num_columns > 20 and 
                    characteristics["size_category"] == "small_dataset"):
                    config = algorithm_map["regularization_needed"]
            
            # Create ML configuration
            ml_config = MLConfig(
                algorithm=config["algorithm"],
                hyperparameters=config["hyperparameters"],
                reasoning=config["reasoning"]
            )
            
            self.logger.info(f"Generated ML config: {ml_config.algorithm}")
            return ml_config
            
        except Exception as e:
            self.logger.error(f"Error generating ML config: {str(e)}")
            raise
    
    def generate_evaluation_config(self, dataset_info: DatasetInfo) -> EvaluationConfig:
        """
        Generate evaluation configuration
        Takes: DatasetInfo
        Returns: EvaluationConfig with recommended evaluation settings
        """
        try:
            pipeline_type = dataset_info.pipeline_type
            
            # Select metrics based on pipeline type
            if pipeline_type == "Classification":
                metrics = ["accuracy", "precision", "recall", "f1"]
                if len(set(dataset_info.column_types.get(dataset_info.target_column, []))) == 2:
                    metrics.append("roc_auc")
            elif pipeline_type == "Regression":
                metrics = ["mae", "mse", "rmse", "r2"]
            elif pipeline_type == "Clustering":
                metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
            else:
                metrics = []
            
            # Adjust cross-validation based on dataset size
            cv_folds = 5
            if dataset_info.num_rows < 500:
                cv_folds = 3
            elif dataset_info.num_rows > 10000:
                cv_folds = 10
            
            evaluation_config = EvaluationConfig(
                metrics=metrics,
                cv_folds=cv_folds
            )
            
            self.logger.info(f"Generated evaluation config with metrics: {metrics}")
            return evaluation_config
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation config: {str(e)}")
            raise
    
    def generate_preprocessing_suggestions(self, dataset_info: DatasetInfo, characteristics: Dict[str, Any]) -> List[str]:
        """
        Generate preprocessing suggestions
        Takes: DatasetInfo and characteristics
        Returns: List of preprocessing suggestions
        """
        suggestions = []
        
        try:
            # Missing data suggestions
            if dataset_info.has_missing_values:
                suggestions.append("Handle missing values before training (imputation or removal)")
            
            # Categorical data suggestions
            if characteristics["categorical_ratio"] > 0.3:
                suggestions.append("Consider encoding categorical variables (one-hot or label encoding)")
            
            # Feature scaling suggestions
            if characteristics["numeric_ratio"] > 0.5:
                suggestions.append("Consider feature scaling (StandardScaler or MinMaxScaler)")
            
            # High dimensionality suggestions
            if characteristics["dimensionality"] == "high_dimensional":
                suggestions.append("Consider dimensionality reduction (PCA or feature selection)")
            
            # Small dataset suggestions
            if characteristics["size_category"] == "small_dataset":
                suggestions.append("Consider cross-validation for robust model evaluation")
                suggestions.append("Watch for overfitting due to small dataset size")
            
            self.logger.info(f"Generated {len(suggestions)} preprocessing suggestions")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating preprocessing suggestions: {str(e)}")
            return []
    
    def generate_warnings(self, dataset_info: DatasetInfo, characteristics: Dict[str, Any]) -> List[str]:
        """
        Generate warnings based on dataset characteristics
        Takes: DatasetInfo and characteristics
        Returns: List of warnings
        """
        warnings = []
        
        try:
            # Small dataset warnings
            if dataset_info.num_rows < 100:
                warnings.append("Very small dataset - results may not be reliable")
            
            # Imbalanced features warning
            if dataset_info.num_columns > dataset_info.num_rows:
                warnings.append("More features than samples - high risk of overfitting")
            
            # Missing target warning
            if dataset_info.pipeline_type != "Clustering" and not dataset_info.target_column:
                warnings.append("No target column specified for supervised learning")
            
            # High missing data warning
            if dataset_info.has_missing_values:
                warnings.append("Dataset contains missing values - preprocessing required")
            
            self.logger.info(f"Generated {len(warnings)} warnings")
            return warnings
            
        except Exception as e:
            self.logger.error(f"Error generating warnings: {str(e)}")
            return []
    
    def generate_config(self, request: ConfigRequest) -> ConfigResponse:
        """
        Generate complete configuration
        Takes: ConfigRequest
        Returns: ConfigResponse with all configurations
        """
        try:
            self.logger.info(f"Generating configuration for project: {request.project_name}")
            
            # Analyze dataset
            characteristics = self.analyze_dataset_characteristics(request.dataset_info)
            
            # Generate configurations
            ml_config = self.generate_ml_config(request.dataset_info, characteristics)
            evaluation_config = self.generate_evaluation_config(request.dataset_info)
            
            # Generate suggestions and warnings
            preprocessing_suggestions = self.generate_preprocessing_suggestions(request.dataset_info, characteristics)
            warnings = self.generate_warnings(request.dataset_info, characteristics)
            
            response = ConfigResponse(
                ml_config=ml_config,
                evaluation_config=evaluation_config,
                preprocessing_suggestions=preprocessing_suggestions,
                warnings=warnings
            )
            
            self.logger.info("Configuration generation completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating configuration: {str(e)}")
            raise

# Initialize generator
config_generator = ConfigGenerator()

# API Endpoints
@app.post("/generate-config", response_model=ConfigResponse)
async def generate_config(request: ConfigRequest):
    """
    Generate ML and evaluation configurations
    Takes: ConfigRequest with dataset info and preferences
    Returns: ConfigResponse with recommended configurations
    """
    try:
        logger.info(f"Received config request for project: {request.project_name}")
        
        # Validate request
        if not request.dataset_info.column_names:
            raise HTTPException(status_code=400, detail="Column names are required")
        
        if request.dataset_info.pipeline_type not in ["Classification", "Regression", "Clustering"]:
            raise HTTPException(status_code=400, detail="Invalid pipeline type")
        
        # Generate configuration
        response = config_generator.generate_config(request)
        
        logger.info(f"Configuration generated successfully for {request.project_name}")
        return response
        
    except Exception as e:
        logger.error(f"Error in generate_config endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns: Health status
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/algorithms/{pipeline_type}")
async def get_available_algorithms(pipeline_type: str):
    """
    Get available algorithms for a pipeline type
    Takes: pipeline_type
    Returns: List of available algorithms
    """
    try:
        if pipeline_type == "Classification":
            algorithms = list(config_generator.CLASSIFICATION_ALGORITHMS.keys())
        elif pipeline_type == "Regression":
            algorithms = list(config_generator.REGRESSION_ALGORITHMS.keys())
        elif pipeline_type == "Clustering":
            algorithms = list(config_generator.CLUSTERING_ALGORITHMS.keys())
        else:
            raise HTTPException(status_code=400, detail="Invalid pipeline type")
        
        return {"pipeline_type": pipeline_type, "algorithms": algorithms}
        
    except Exception as e:
        logger.error(f"Error getting algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
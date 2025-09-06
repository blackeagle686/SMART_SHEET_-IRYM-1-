import json
import requests
import logging
from typing import Dict, Any, Optional
from django.conf import settings
from .models import MLProject, DatasetInfo, MLConfiguration

logger = logging.getLogger(__name__)

class ConfigSender:
    """
    Sender class to communicate with FastAPI config generator
    Sends dataset information and receives ML configurations
    """
    
    def __init__(self, fastapi_base_url: str = "http://localhost:8000"):
        """
        Initialize config sender
        Takes: FastAPI base URL
        Returns: ConfigSender instance
        """
        self.base_url = fastapi_base_url.rstrip('/')
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def prepare_dataset_info(self, project: MLProject, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """
        Prepare dataset information for FastAPI request
        Takes: MLProject and DatasetInfo instances
        Returns: Dictionary with dataset information
        """
        try:
            dataset_data = {
                "num_rows": dataset_info.num_rows,
                "num_columns": dataset_info.num_columns,
                "column_names": dataset_info.column_names,
                "column_types": dataset_info.column_types,
                "target_column": dataset_info.target_column,
                "has_missing_values": dataset_info.has_missing_values,
                "pipeline_type": project.pipeline_type
            }
            
            self.logger.info(f"Prepared dataset info for project {project.id}")
            return dataset_data
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset info: {str(e)}")
            raise
    
    def send_config_request(self, project: MLProject, dataset_info: DatasetInfo, 
                          user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send configuration request to FastAPI service
        Takes: MLProject, DatasetInfo, optional user preferences
        Returns: Generated configuration from FastAPI
        """
        try:
            # Prepare request data
            dataset_data = self.prepare_dataset_info(project, dataset_info)
            
            request_data = {
                "dataset_info": dataset_data,
                "user_preferences": user_preferences or {},
                "project_name": project.name,
                "project_description": project.description or ""
            }
            
            # Send request to FastAPI
            url = f"{self.base_url}/generate-config"
            
            self.logger.info(f"Sending config request to {url}")
            
            response = self.session.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response.raise_for_status()
            
            config_data = response.json()
            
            self.logger.info(f"Received config response for project {project.id}")
            return config_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise Exception(f"Failed to communicate with config service: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error sending config request: {str(e)}")
            raise
    
    def get_available_algorithms(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Get available algorithms for a pipeline type
        Takes: pipeline_type (Classification, Regression, Clustering)
        Returns: Dictionary with available algorithms
        """
        try:
            url = f"{self.base_url}/algorithms/{pipeline_type}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            algorithms_data = response.json()
            
            self.logger.info(f"Retrieved algorithms for {pipeline_type}")
            return algorithms_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error getting algorithms: {str(e)}")
            raise Exception(f"Failed to get algorithms: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error getting algorithms: {str(e)}")
            raise
    
    def check_service_health(self) -> bool:
        """
        Check if FastAPI service is healthy
        Takes: None
        Returns: Boolean indicating service health
        """
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            health_data = response.json()
            is_healthy = health_data.get("status") == "healthy"
            
            self.logger.info(f"Service health check: {is_healthy}")
            return is_healthy
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            return False

class ConfigReceiver:
    """
    Receiver class to process configurations from FastAPI
    Processes and stores ML configurations in Django models
    """
    
    def __init__(self):
        """
        Initialize config receiver
        """
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_ml_config(self, project: MLProject, ml_config_data: Dict[str, Any]) -> MLConfiguration:
        """
        Process and save ML configuration from FastAPI response
        Takes: MLProject instance and ML config data
        Returns: Created MLConfiguration instance
        """
        try:
            # Extract ML config data
            ml_config = ml_config_data.get('ml_config', {})
            
            # Create or update MLConfiguration
            config, created = MLConfiguration.objects.get_or_create(
                project=project,
                defaults={
                    'algorithm': ml_config.get('algorithm', ''),
                    'hyperparameters': ml_config.get('hyperparameters', {}),
                    'validation_split': ml_config.get('validation_split', 0.2),
                    'cross_validation_folds': ml_config.get('cross_validation_folds', 5),
                    'random_seed': ml_config.get('random_seed', 42),
                    'use_cross_validation': ml_config.get('use_cross_validation', True)
                }
            )
            
            if not created:
                # Update existing configuration
                config.algorithm = ml_config.get('algorithm', config.algorithm)
                config.hyperparameters = ml_config.get('hyperparameters', config.hyperparameters)
                config.validation_split = ml_config.get('validation_split', config.validation_split)
                config.cross_validation_folds = ml_config.get('cross_validation_folds', config.cross_validation_folds)
                config.random_seed = ml_config.get('random_seed', config.random_seed)
                config.use_cross_validation = ml_config.get('use_cross_validation', config.use_cross_validation)
                config.save()
            
            self.logger.info(f"Processed ML config for project {project.id}: {config.algorithm}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error processing ML config: {str(e)}")
            raise
    
    def process_evaluation_config(self, evaluation_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process evaluation configuration from FastAPI response
        Takes: evaluation config data
        Returns: Processed evaluation configuration
        """
        try:
            eval_config = evaluation_config_data.get('evaluation_config', {})
            
            processed_config = {
                'metrics': eval_config.get('metrics', []),
                'generate_plots': eval_config.get('generate_plots', True),
                'include_feature_importance': eval_config.get('include_feature_importance', True),
                'export_formats': eval_config.get('export_formats', ['json', 'html']),
                'use_cross_validation': eval_config.get('use_cross_validation', True),
                'cv_folds': eval_config.get('cv_folds', 5)
            }
            
            self.logger.info(f"Processed evaluation config with metrics: {processed_config['metrics']}")
            return processed_config
            
        except Exception as e:
            self.logger.error(f"Error processing evaluation config: {str(e)}")
            raise
    
    def process_suggestions_and_warnings(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process preprocessing suggestions and warnings
        Takes: complete config data
        Returns: Dictionary with suggestions and warnings
        """
        try:
            suggestions = config_data.get('preprocessing_suggestions', [])
            warnings = config_data.get('warnings', [])
            
            processed_data = {
                'suggestions': suggestions,
                'warnings': warnings,
                'reasoning': config_data.get('ml_config', {}).get('reasoning', ''),
                'timestamp': config_data.get('timestamp', '')
            }
            
            self.logger.info(f"Processed {len(suggestions)} suggestions and {len(warnings)} warnings")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing suggestions and warnings: {str(e)}")
            raise
    
    def process_complete_config(self, project: MLProject, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete configuration response from FastAPI
        Takes: MLProject instance and complete config data
        Returns: Dictionary with all processed configurations
        """
        try:
            self.logger.info(f"Processing complete config for project {project.id}")
            
            # Process ML configuration
            ml_config = self.process_ml_config(project, config_data)
            
            # Process evaluation configuration
            eval_config = self.process_evaluation_config(config_data)
            
            # Process suggestions and warnings
            suggestions_warnings = self.process_suggestions_and_warnings(config_data)
            
            result = {
                'ml_configuration': ml_config,
                'evaluation_configuration': eval_config,
                'suggestions_and_warnings': suggestions_warnings,
                'success': True
            }
            
            self.logger.info(f"Successfully processed complete config for project {project.id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing complete config: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class MLConfigurationService:
    """
    Service class that combines sender and receiver for complete ML configuration workflow
    """
    
    def __init__(self, fastapi_url: str = "http://localhost:8000"):
        """
        Initialize ML configuration service
        Takes: FastAPI service URL
        Returns: MLConfigurationService instance
        """
        self.sender = ConfigSender(fastapi_url)
        self.receiver = ConfigReceiver()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_and_apply_config(self, project: MLProject, dataset_info: DatasetInfo,
                                 user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate configuration using FastAPI and apply it to Django models
        Takes: MLProject, DatasetInfo, optional user preferences
        Returns: Complete processing results
        """
        try:
            self.logger.info(f"Starting config generation for project {project.id}")
            
            # Check service health first
            if not self.sender.check_service_health():
                raise Exception("Configuration service is not available")
            
            # Send request to FastAPI
            config_data = self.sender.send_config_request(project, dataset_info, user_preferences)
            
            # Process response and update Django models
            result = self.receiver.process_complete_config(project, config_data)
            
            if result['success']:
                self.logger.info(f"Successfully generated and applied config for project {project.id}")
            else:
                self.logger.error(f"Failed to process config for project {project.id}: {result.get('error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generate_and_apply_config: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_algorithm_recommendations(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Get algorithm recommendations for a pipeline type
        Takes: pipeline_type
        Returns: Algorithm recommendations
        """
        try:
            return self.sender.get_available_algorithms(pipeline_type)
        except Exception as e:
            self.logger.error(f"Error getting algorithm recommendations: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_service_connection(self) -> Dict[str, Any]:
        """
        Validate connection to FastAPI service
        Takes: None
        Returns: Connection status
        """
        try:
            is_healthy = self.sender.check_service_health()
            return {
                'connected': is_healthy,
                'service_url': self.sender.base_url,
                'status': 'healthy' if is_healthy else 'unhealthy'
            }
        except Exception as e:
            self.logger.error(f"Error validating service connection: {str(e)}")
            return {
                'connected': False,
                'error': str(e),
                'service_url': self.sender.base_url,
                'status': 'error'
            }

# Utility functions for easy integration
def generate_ml_config(project: MLProject, dataset_info: DatasetInfo, 
                      user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Utility function to generate ML configuration
    Takes: MLProject, DatasetInfo, optional user preferences
    Returns: Configuration generation results
    """
    service = MLConfigurationService()
    return service.generate_and_apply_config(project, dataset_info, user_preferences)

def check_config_service() -> bool:
    """
    Utility function to check if configuration service is available
    Takes: None
    Returns: Boolean indicating service availability
    """
    sender = ConfigSender()
    return sender.check_service_health()

def get_algorithm_options(pipeline_type: str) -> Dict[str, Any]:
    """
    Utility function to get algorithm options
    Takes: pipeline_type
    Returns: Available algorithms
    """
    sender = ConfigSender()
    try:
        return sender.get_available_algorithms(pipeline_type)
    except Exception:
        return {'algorithms': []}
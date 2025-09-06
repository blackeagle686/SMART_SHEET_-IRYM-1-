from django.db import models
from django.contrib.auth.models import User
import json
from sheet_models.models import Analysis, ML_Models

class MLProject(models.Model):
    """
    Model to store ML project information
    """
    PIPELINE_TYPES = [
        ('Classification', 'Classification'),
        ('Regression', 'Regression'),
        ('Clustering', 'Clustering'),
    ]
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name='ml_projects')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    pipeline_type = models.CharField(max_length=20, choices=PIPELINE_TYPES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def __str__(self):
        return f"{self.name} - {self.pipeline_type}"

class DatasetInfo(models.Model):
    """
        Model to store dataset information
    """
    project = models.OneToOneField(MLProject, on_delete=models.CASCADE)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()
    num_rows = models.IntegerField()
    num_columns = models.IntegerField()
    column_names = models.JSONField()  # Store column names as JSON array
    column_types = models.JSONField()  # Store column types as JSON dict
    target_column = models.CharField(max_length=100, blank=True)
    has_missing_values = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Dataset for {self.project.name}"

class MLConfiguration(models.Model):
    """
        Model to store ML training configuration
    """
    project = models.OneToOneField(MLProject, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=100)
    hyperparameters = models.JSONField(default=dict)
    validation_split = models.FloatField(default=0.2)
    cross_validation_folds = models.IntegerField(default=5)
    random_seed = models.IntegerField(default=42)
    use_cross_validation = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Config for {self.project.name}"

class TrainingResult(models.Model):
    """
    Model to store training results
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    project = models.ForeignKey(MLProject, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    model_path = models.CharField(max_length=500, blank=True)
    metrics = models.JSONField(default=dict)
    training_time = models.FloatField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Training for {self.project.name} - {self.status}"

class EvaluationResult(models.Model):
    """
    Model to store evaluation results
    """
    training_result = models.OneToOneField(TrainingResult, on_delete=models.CASCADE)
    evaluation_metrics = models.JSONField(default=dict)
    plots_data = models.JSONField(default=dict)  # Store plot file paths
    confusion_matrix = models.JSONField(null=True, blank=True)
    classification_report = models.JSONField(null=True, blank=True)
    feature_importance = models.JSONField(default=dict)
    evaluation_time = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Evaluation for {self.training_result.project.name}"

class PredictionRequest(models.Model):
    """
    Model to store prediction requests
    """
    project = models.ForeignKey(MLProject, on_delete=models.CASCADE)
    input_data = models.JSONField()  # Store input features as JSON
    prediction_result = models.JSONField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.project.name} at {self.created_at}"

class SystemLog(models.Model):
    """
    Model to store system logs
    """
    LOG_LEVELS = [
        ('INFO', 'Info'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
        ('DEBUG', 'Debug'),
    ]
    
    project = models.ForeignKey(MLProject, on_delete=models.CASCADE, null=True, blank=True)
    level = models.CharField(max_length=10, choices=LOG_LEVELS)
    message = models.TextField()
    function_name = models.CharField(max_length=100, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    extra_data = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.level} - {self.message[:50]}..."
    
    
class ModerWheights(models.Model):
    model_name = models.ForeignKey(ML_Models, on_delete=models.CASCADE, related_name='model_weights')
    weights_file = models.FileField(upload_to='model_weights/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"Weights for {self.model_name} uploaded at {self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}"
    
    
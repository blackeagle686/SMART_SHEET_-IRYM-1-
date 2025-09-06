from django.db import models
from django.contrib.auth.models import User
# from .data_science_unit import MODELS
from django.utils import timezone

MODELS = {
    "Logistic Regression": "LogisticRegression",
    "Linear Regression": "LinearRegression",
    "Ridge Regression": "Ridge",
    "Lasso Regression": "Lasso",
    "Decision Tree Classifier": "DecisionTreeClassifier",
    "Decision Tree Regressor": "DecisionTreeRegressor",
    "Random Forest Classifier": "RandomForestClassifier",
    "Random Forest Regressor": 'RandomForestRegressor',
    "Gradient Boosting Classifier": "GradientBoostingClassifier",
    "Gradient Boosting Regressor": 'GradientBoostingRegressor',
    "AdaBoost Classifier": 'AdaBoostClassifier',
    "AdaBoost Regressor": 'AdaBoostRegressor',
    "Support Vector Classifier": 'SVC',
    "Support Vector Regressor": 'SVR',
    "Gaussian Naive Bayes": 'GaussianNB',
    "KNN Classifier": 'KNeighborsClassifier',
    "KNN Regressor": 'KNeighborsRegressor',
    "KMeans Clustering": 'KMeans',
}

class Prompt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="user_prompts")
    prompt = models.TextField(max_length=4000)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"user {self.user.username} created this prompt at {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"

class RawData(models.Model):
    prompt = models.ForeignKey(
        Prompt,
        on_delete=models.CASCADE,
        related_name="raw_data"
    )
    data = models.FileField(
        upload_to='raw_data_files/',  
        help_text="Upload the raw data file"
    )
    size = models.PositiveIntegerField(
        help_text="Size of the data in bytes"
    )
    type = models.CharField(
        max_length=20,
        choices=[
            ("csv", "CSV"),
            ("excel", "Excel"),
            ("json", "JSON"),
            ("xml", "XML"),
        ],
        default="csv",
        help_text="Type/format of the raw data file"
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the file was uploaded"
    )

    def __str__(self):
        return f"RawData ({self.type}) linked to Prompt ID {self.prompt.id}"

class Analysis(models.Model):
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE, related_name="prompt_analyses")
    title = models.CharField(max_length=100)
    description = models.TextField(max_length=2000)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis by {self.prompt.user.username} at {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"

class CleanedData(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="analysis_cleaned_data")
    clean_data = models.FileField(upload_to='cleaned_data_files/')
    target_column = models.CharField(max_length=100, help_text="Name of the target column")
    
    scaller = models.CharField(max_length=100, null=True, blank=True, help_text="Scaler used for feature scaling")
    normalier = models.CharField(max_length=100, null=True, blank=True, help_text="Normalizer used for feature normalization")
    encoder = models.CharField(max_length=100, null=True, blank=True, help_text="Encoder used for categorical variables")
    
    size = models.PositiveIntegerField(help_text="Size of the cleaned data in bytes")
    type = models.CharField(
        max_length=20,
        choices=[
            ("csv", "CSV"),
            ("excel", "Excel"),
            ("json", "JSON"),
            ("xml", "XML"),
        ],
        default="csv",
        help_text="Type/format of the cleaned data file"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CleanedData ({self.type}) for Analysis ID {self.analysis.id}"

class ML_Models(models.Model):
    ml_models = [(k, k) for k in MODELS.keys()]

    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="analysis_ml_models")
    model = models.CharField(choices=ml_models, max_length=200)
    description = models.TextField(max_length=2000)
    model_type = models.CharField(choices=[
        ("classification", "Classification"),
        ("regression", "Regression"),
        ("clustering", "Clustering")
    ], max_length=20, null=True, blank=True)
    
    def __str__(self):
        return f"Model '{self.model}' belongs to Analysis ID: {self.analysis.id}"

class ModelHyperParameters(models.Model):
    ml_model = models.ForeignKey(
        'ML_Models',  
        on_delete=models.CASCADE,
        related_name='hyperparameters'  
        
    )
    hyperparams = models.JSONField(help_text="Hyperparameters as a JSON object", null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now , null=True, blank=True)    
    def __str__(self):
        return f"Hyperparameters for Model ID: {self.ml_model.id} at {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"


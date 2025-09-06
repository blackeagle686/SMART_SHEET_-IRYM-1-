from django import forms
from .models import MLProject, DatasetInfo, MLConfiguration, PredictionRequest
import json
import pandas as pd
import numpy as np
from sheet_models.models import CleanedData, ML_Models, ModelHyperParameters

class PredictionForm(forms.Form):  # لو BaseForm، اكتب BaseForm بدل forms.Form
    """
    فورم ديناميكي بدون kwargs إضافية.
    """
    pass  # فارغ عند الإنشاء


class TrainingConfigForm(forms.Form):
    target_column = forms.CharField(
        label="Target Column",
        required=True,
        widget=forms.TextInput(attrs={"class": "form-control"})
    )

    ml_task = forms.ChoiceField(
        label="ML Task",
        required=True,
        choices=[
            ("classification", "Classification"),
            ("regression", "Regression"),
            ("clustering", "Clustering"),
        ],
        widget=forms.Select(attrs={"class": "form-select"})
    )

    ml_algo = forms.CharField(
        label="ML Algorithm",
        required=True,
        widget=forms.TextInput(attrs={"class": "form-control"})
    )

    hyperparams = forms.CharField(
        label="Hyperparameters (JSON)",
        required=False,
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 5})
    )

    def clean_hyperparams(self):
        """تأكد أن الهايبر باراميترز JSON صح"""
        data = self.cleaned_data.get("hyperparams")
        if not data:
            return {}
        try:
            return json.loads(data)
        except Exception:
            raise forms.ValidationError("Hyperparameters must be a valid JSON object.")

class HyperParameterForm(forms.Form):
    def __init__(self, *args, hyperparams=None, **kwargs):
        super().__init__(*args, **kwargs)

        if hyperparams:
            for key, value in hyperparams.items():
                # int
                if isinstance(value, int):
                    self.fields[key] = forms.IntegerField(
                        label=key,
                        initial=value,
                        required=False
                    )
                # float
                elif isinstance(value, float):
                    self.fields[key] = forms.FloatField(
                        label=key,
                        initial=value,
                        required=False
                    )
                # bool
                elif isinstance(value, bool):
                    self.fields[key] = forms.BooleanField(
                        label=key,
                        initial=value,
                        required=False
                    )
                # string أو أي حاجة تانية
                else:
                    self.fields[key] = forms.CharField(
                        label=key,
                        initial=value,
                        required=False
                    )

class MLProjectForm(forms.ModelForm):
    """
    Form for creating and editing ML projects
    Takes: Project name, description, pipeline type
    Returns: MLProject instance
    """
    class Meta:
        model = MLProject
        fields = ['name', 'description', 'pipeline_type']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter project name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Project description (optional)'
            }),
            'pipeline_type': forms.Select(attrs={
                'class': 'form-select'
            })
        }

class DatasetUploadForm(forms.ModelForm):
    """
    Form for uploading dataset information
    Takes: Dataset file, target column
    Returns: DatasetInfo instance
    """
    dataset_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.pkl,.npy'
        }),
        help_text='Upload CSV, PKL, or NPY file'
    )
    
    class Meta:
        model = DatasetInfo
        fields = ['target_column']
        widgets = {
            'target_column': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Target column name (leave empty for clustering)'
            })
        }

class MLConfigurationForm(forms.ModelForm):
    """
    Form for ML algorithm configuration
    Takes: Algorithm selection and hyperparameters
    Returns: MLConfiguration instance
    """
    
    # Algorithm choices based on pipeline type
    CLASSIFICATION_ALGORITHMS = [
        ('Logistic Regression', 'Logistic Regression'),
        ('Random Forest Classifier', 'Random Forest Classifier'),
        ('Decision Tree Classifier', 'Decision Tree Classifier'),
        ('Support Vector Classifier', 'Support Vector Classifier'),
        ('Gaussian Naive Bayes', 'Gaussian Naive Bayes'),
        ('KNN Classifier', 'KNN Classifier'),
        ('Gradient Boosting Classifier', 'Gradient Boosting Classifier'),
        ('AdaBoost Classifier', 'AdaBoost Classifier'),
    ]
    
    REGRESSION_ALGORITHMS = [
        ('Linear Regression', 'Linear Regression'),
        ('Ridge Regression', 'Ridge Regression'),
        ('Lasso Regression', 'Lasso Regression'),
        ('Random Forest Regressor', 'Random Forest Regressor'),
        ('Decision Tree Regressor', 'Decision Tree Regressor'),
        ('Support Vector Regressor', 'Support Vector Regressor'),
        ('KNN Regressor', 'KNN Regressor'),
        ('Gradient Boosting Regressor', 'Gradient Boosting Regressor'),
        ('AdaBoost Regressor', 'AdaBoost Regressor'),
    ]
    
    CLUSTERING_ALGORITHMS = [
        ('KMeans Clustering', 'KMeans Clustering'),
    ]
    
    algorithm = forms.ChoiceField(
        choices=[],  # Will be populated dynamically
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Hyperparameter fields
    n_estimators = forms.IntegerField(
        required=False,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'min': 1})
    )
    
    max_depth = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'min': 1})
    )
    
    learning_rate = forms.FloatField(
        required=False,
        initial=0.1,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0.01'})
    )
    
    n_clusters = forms.IntegerField(
        required=False,
        initial=3,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'min': 2})
    )
    
    c_parameter = forms.FloatField(
        required=False,
        initial=1.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0.1'})
    )
    
    n_neighbors = forms.IntegerField(
        required=False,
        initial=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'min': 1})
    )
    
    class Meta:
        model = MLConfiguration
        fields = ['algorithm', 'validation_split', 'cross_validation_folds', 'random_seed', 'use_cross_validation']
        widgets = {
            'validation_split': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0.1',
                'max': '0.9'
            }),
            'cross_validation_folds': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '2',
                'max': '10'
            }),
            'random_seed': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1'
            }),
            'use_cross_validation': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
    
    def __init__(self, *args, **kwargs):
        pipeline_type = kwargs.pop('pipeline_type', None)
        super().__init__(*args, **kwargs)
        
        # Set algorithm choices based on pipeline type
        if pipeline_type == 'Classification':
            self.fields['algorithm'].choices = self.CLASSIFICATION_ALGORITHMS
        elif pipeline_type == 'Regression':
            self.fields['algorithm'].choices = self.REGRESSION_ALGORITHMS
        elif pipeline_type == 'Clustering':
            self.fields['algorithm'].choices = self.CLUSTERING_ALGORITHMS
    
    def clean(self):
        """
        Custom validation to build hyperparameters dict
        """
        cleaned_data = super().clean()
        algorithm = cleaned_data.get('algorithm')
        
        # Build hyperparameters dict based on algorithm
        hyperparameters = {}
        
        if algorithm in ['Random Forest Classifier', 'Random Forest Regressor']:
            if cleaned_data.get('n_estimators'):
                hyperparameters['n_estimators'] = cleaned_data['n_estimators']
            if cleaned_data.get('max_depth'):
                hyperparameters['max_depth'] = cleaned_data['max_depth']
        
        elif algorithm in ['Gradient Boosting Classifier', 'Gradient Boosting Regressor']:
            if cleaned_data.get('n_estimators'):
                hyperparameters['n_estimators'] = cleaned_data['n_estimators']
            if cleaned_data.get('learning_rate'):
                hyperparameters['learning_rate'] = cleaned_data['learning_rate']
        
        elif algorithm in ['Support Vector Classifier', 'Support Vector Regressor']:
            if cleaned_data.get('c_parameter'):
                hyperparameters['C'] = cleaned_data['c_parameter']
        
        elif algorithm in ['KNN Classifier', 'KNN Regressor']:
            if cleaned_data.get('n_neighbors'):
                hyperparameters['n_neighbors'] = cleaned_data['n_neighbors']
        
        elif algorithm == 'KMeans Clustering':
            if cleaned_data.get('n_clusters'):
                hyperparameters['n_clusters'] = cleaned_data['n_clusters']
        
        cleaned_data['hyperparameters'] = hyperparameters
        return cleaned_data

class PredictionForm(forms.Form):
    """
    Dynamic form for making predictions
    Takes: Feature values based on dataset columns
    Returns: Input data for prediction
    """
    
    def __init__(self, *args, **kwargs):
        dataset_info = kwargs.pop('dataset_info', None)
        super().__init__(*args, **kwargs)
        
        if dataset_info:
            # Create form fields based on dataset columns
            column_names = dataset_info.column_names
            column_types = dataset_info.column_types
            target_column = dataset_info.target_column
            
            for column in column_names:
                if column != target_column:  # Skip target column
                    field_type = column_types.get(column, 'object')
                    
                    if field_type in ['int64', 'int32', 'int16', 'int8']:
                        self.fields[column] = forms.IntegerField(
                            label=column,
                            required=True,
                            widget=forms.NumberInput(attrs={'class': 'form-control'})
                        )
                    elif field_type in ['float64', 'float32', 'float16']:
                        self.fields[column] = forms.FloatField(
                            label=column,
                            required=True,
                            widget=forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'})
                        )
                    else:  # String or object type
                        self.fields[column] = forms.CharField(
                            label=column,
                            required=True,
                            widget=forms.TextInput(attrs={'class': 'form-control'})
                        )

class EvaluationConfigForm(forms.Form):
    """
    Form for evaluation pipeline configuration
    Takes: Evaluation settings and metrics selection
    Returns: Evaluation configuration dict
    """
    
    # Metrics choices
    CLASSIFICATION_METRICS = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1-Score'),
        ('roc_auc', 'ROC AUC'),
    ]
    
    REGRESSION_METRICS = [
        ('mae', 'Mean Absolute Error'),
        ('mse', 'Mean Squared Error'),
        ('rmse', 'Root Mean Squared Error'),
        ('r2', 'R² Score'),
        ('mape', 'Mean Absolute Percentage Error'),
    ]
    
    CLUSTERING_METRICS = [
        ('silhouette', 'Silhouette Score'),
        ('davies_bouldin', 'Davies Bouldin Score'),
        ('calinski_harabasz', 'Calinski Harabasz Score'),
    ]
    
    use_cross_validation = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    cv_folds = forms.IntegerField(
        initial=5,
        min_value=2,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    generate_plots = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    include_feature_importance = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    export_formats = forms.MultipleChoiceField(
        choices=[
            ('json', 'JSON'),
            ('html', 'HTML'),
            ('markdown', 'Markdown'),
        ],
        initial=['json', 'html'],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    
    def __init__(self, *args, **kwargs):
        pipeline_type = kwargs.pop('pipeline_type', None)
        super().__init__(*args, **kwargs)
        
        # Add metrics field based on pipeline type
        if pipeline_type == 'Classification':
            self.fields['metrics'] = forms.MultipleChoiceField(
                choices=self.CLASSIFICATION_METRICS,
                initial=['accuracy', 'precision', 'recall', 'f1'],
                widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
            )
        elif pipeline_type == 'Regression':
            self.fields['metrics'] = forms.MultipleChoiceField(
                choices=self.REGRESSION_METRICS,
                initial=['mae', 'mse', 'r2'],
                widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
            )
        elif pipeline_type == 'Clustering':
            self.fields['metrics'] = forms.MultipleChoiceField(
                choices=self.CLUSTERING_METRICS,
                initial=['silhouette', 'davies_bouldin'],
                widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
            )

class FileUploadForm(forms.Form):
    """
    Simple file upload form
    Takes: File upload
    Returns: Uploaded file
    """
    file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.pkl,.npy'
        })
    )
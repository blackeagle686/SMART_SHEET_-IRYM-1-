from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import json
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

from .models import (
    MLProject, DatasetInfo, MLConfiguration, ModerWheights,
    TrainingResult, EvaluationResult, PredictionRequest, SystemLog
)
from .forms import (
    MLProjectForm, DatasetUploadForm, MLConfigurationForm, 
    PredictionForm, EvaluationConfigForm, FileUploadForm
)
from .ml_pipeline import TrainingPipeline
from .evaluation_pipeline import EvaluationPipeline
from .config_generator import ConfigGenerator
from sheet_models.data_collector import GetData
# from .utils import setup_project_logging
from sheet_models.models import Analysis,CleanedData, ML_Models, ModelHyperParameters


# from django.shortcuts import render, redirect, get_object_or_404
# from django.contrib.auth.decorators import login_required
# from django.http import HttpResponseForbidden, HttpResponse
# from django.contrib import messages
# import logging
# from .forms import PredictionForm
# from sheet_models.models import Analysis, CleanedData, ML_Models, ModelHyperParameters
# from sheet_models.data_collector import GetData
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
# from sklearn.svm import SVC, SVR
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.naive_bayes import GaussianNB
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# import joblib
# import pandas as pd
# import numpy as np
# import json
# import os
# try:
#     from xgboost import XGBClassifier, XGBRegressor
#     from lightgbm import LGBMClassifier, LGBMRegressor
# except ImportError:
#     XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor = None, None, None, None

# # Dictionary لربط أسماء الخوارزميات بالكائنات
# ALGORITHM_MAP = {
#     'classification': {
#         'Random Forest Classifier': RandomForestClassifier,
#         'Logistic Regression': LogisticRegression,
#         'Support Vector Classifier': SVC,
#         'KNN Classifier': KNeighborsClassifier,
#         'Decision Tree Classifier': DecisionTreeClassifier,
#         'Gradient Boosting Classifier': GradientBoostingClassifier,
#         'AdaBoost Classifier': AdaBoostClassifier,
#         'Gaussian Naive Bayes': GaussianNB,
#         'XGBoost': XGBClassifier if XGBClassifier else None,
#         'LightGBM': LGBMClassifier if LGBMClassifier else None,
#     },
#     'regression': {
#         'Random Forest Regressor': RandomForestRegressor,
#         'Linear Regression': LinearRegression,
#         'Support Vector Regressor': SVR,
#         'KNN Regressor': KNeighborsRegressor,
#         'Decision Tree Regressor': DecisionTreeRegressor,
#         'Gradient Boosting Regressor': GradientBoostingRegressor,
#         'AdaBoost Regressor': AdaBoostRegressor,
#         'ElasticNet': ElasticNet,
#         'XGBoost': XGBRegressor if XGBRegressor else None,
#         'LightGBM': LGBMRegressor if LGBMRegressor else None,
#     },
#     'clustering': {
#         'KMeans Clustering': KMeans,
#         'DBSCAN': DBSCAN,
#     }
# }


# # Setup logger
# logger = logging.getLogger(__name__)

# def log_system_event(project, level, message, function_name="", extra_data=None):
#     """
#     Log system events to database
#     Takes: project instance, log level, message, function name, extra data
#     Returns: SystemLog instance
#     """
#     try:
#         log_entry = SystemLog.objects.create(
#             project=project,
#             level=level,
#             message=message,
#             function_name=function_name,
#             extra_data=extra_data or {}
#         )
#         return log_entry
#     except Exception as e:
#         logger.error(f"Failed to log system event: {str(e)}")

# # @login_required
# # def ml_model_result(request, analysis_id):
# #     try:
# #         # استرجاع التحليل والتأكد من صلاحية المستخدم
# #         analysis = get_object_or_404(Analysis, id=analysis_id)
# #         if analysis.prompt.user != request.user:
# #             return HttpResponseForbidden("Not allowed BadGate")

# #         # استرجاع البيانات المرتبطة
# #         data_file = get_object_or_404(CleanedData, analysis=analysis)
# #         ml_model_info = get_object_or_404(ML_Models, analysis=analysis)
# #         hyperparams = get_object_or_404(ModelHyperParameters, ml_model=ml_model_info)

# #         # استخراج معلومات النموذج والبيانات
# #         ml_task = ml_model_info.model_type
# #         ml_algo = ml_model_info.model
# #         ml_hyperparams = hyperparams.hyperparams
# #         scaller = data_file.scaller
# #         normalier = data_file.normalier
# #         encoder = data_file.encoder
# #         data_path = data_file.clean_data.path

# #         # جلب بيانات التدريب
# #         data_provider = GetData(cleaned_data_path=data_path, cleaned_type=data_file.type, target=data_file.target_column)
# #         try:
# #             from sklearn.model_selection import train_test_split
# #             train_info = data_provider.get_train_data()
# #             X_train, X_test, y_train, y_test = train_test_split(
# #                 train_info['X'], train_info['y'], test_size=0.2, random_state=42
# #             )
            
# #         except Exception as e:
# #             print("Error in getting train data:", str(e))
# #             logger.error(f"Error in getting train data: {str(e)}")
# #             return render(request, 'pages/ml-details.html', {'error': str(e)})

# #         # تحويل الـ hyperparameters إلى dict إذا كانت JSON string
# #         if isinstance(ml_hyperparams, str):
# #             ml_hyperparams = json.loads(ml_hyperparams)

# #         # تدريب النموذج بناءً على الخوارزمية
# #         model = None
# #         if ml_algo == "RandomForest":
# #             model = RandomForestClassifier(**ml_hyperparams)
# #         elif ml_algo == "LogisticRegression":
# #             model = LogisticRegression(**ml_hyperparams)
# #         elif ml_algo == "SVM":
# #             model = SVC(**ml_hyperparams)
# #         else:
# #             raise ValueError(f"Unsupported algorithm: {ml_algo}")

# #         # تدريب النموذج
# #         model.fit(X_train, y_train)

# #         # حساب الدقة
# #         predictions = model.predict(X_test)
# #         accuracy = accuracy_score(y_test, predictions)

# #         # إعداد بيانات مخطط التقييم (Confusion Matrix)
# #         cm = confusion_matrix(y_test, predictions)
# #         chart_data = {
# #             'labels': [f"Class {i}" for i in range(len(cm))],
# #             'data': cm.tolist()
# #         }

# #         # حفظ أوزان النموذج
# #         weights_path = os.path.join('weights', f'model_{analysis_id}.pkl')
# #         os.makedirs(os.path.dirname(weights_path), exist_ok=True)
# #         joblib.dump(model, weights_path)
# #         ModerWheights.objects.create(model_name=ml_model_info, weights_file=weights_path)

# #         # إعداد الفورم للتوقعات
# #         feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(X_train.shape[1])]
# #         form = PredictionForm(feature_names=feature_names)
# #         prediction_result = None

# #         # إضافة أنواع الأعمدة (categorical/numerical) علشان تتعرض في الـ template
# #         form_fields = []
# #         for field_name, field in form.fields.items():
# #             field_type = "Numeric" if isinstance(X_train, pd.DataFrame) and pd.api.types.is_numeric_dtype(X_train[field_name]) else "Categorical"
# #             form_fields.append({
# #                 "field": form[field_name],   # Field نفسه
# #                 "name": field_name,          # اسم العمود
# #                 "type": field_type           # النوع
# #             })
            
# #         context = {
# #             'ml_task': ml_task,
# #             'ml_algo': ml_algo,
# #             'ml_hyperparams': json.dumps(ml_hyperparams, indent=2),
# #             'data_path': data_path,
# #             'scaller': scaller or "None",
# #             'normalier': normalier or "None",
# #             'encoder': encoder or "None",
# #             'accuracy': round(accuracy * 100, 2),
# #             'chart_data': json.dumps(chart_data),
# #             'form': form,
# #             'prediction_result': prediction_result,
# #             'form_fields': form_fields,
# #             'prediction_result': prediction_result,        }
# #         return render(request, 'pages/ml-details.html', context)

# #     except Exception as e:
# #         print('=======================================================')
# #         print("error", str(e))
# #         print('=======================================================')
# #         logger.error(f"Error in ml_model_result: {str(e)}")
# #         messages.error(request, f"Error loading results: {str(e)}")
# #         return render(request, 'pages/ml-details.html', {'error': str(e)})

# def train_model(ml_task, ml_algo, ml_hyperparams, X, y):
#     """
#     تدريب النموذج بناءً على نوع المهمة والخوارزمية.
    
#     Args:
#         ml_task (str): نوع المهمة (classification, regression, clustering)
#         ml_algo (str): اسم الخوارزمية
#         ml_hyperparams (dict): الـ hyperparameters للنموذج
#         X: بيانات التدريب (المتغيرات المستقلة)
#         y: بيانات التدريب (المتغير الهدف)
    
#     Returns:
#         model: النموذج المدرب
#     """
#     hypers = {}
#     for key, value in ml_hyperparams.items():
#         if isinstance(value, list):
#             # اختر أول قيمة ليست None
#             value = next((v for v in value if v is not None), None)
#         # حاول تحويلها إلى int أو float
#         try:
#             value = int(value)
#         except (ValueError, TypeError):
#             try:
#                 value = float(value)
#             except (ValueError, TypeError):
#                 pass
#         hypers[key] = value

#     ml_hyperparams = hypers


    
    
#     # التحقق من صحة المهمة والخوارزمية
#     if ml_task not in ALGORITHM_MAP:
#         raise ValueError(f"Unsupported ML task: {ml_task}")
#     if ml_algo not in ALGORITHM_MAP[ml_task]:
#         raise ValueError(f"Unsupported algorithm: {ml_algo} for task: {ml_task}")
#     if ALGORITHM_MAP[ml_task][ml_algo] is None:
#         raise ValueError(f"Algorithm {ml_algo} is not available. Please ensure required libraries are installed.")

#     # إنشاء النموذج
#     model_class = ALGORITHM_MAP[ml_task][ml_algo]
#     model = model_class(**ml_hyperparams)

#     # تقسيم البيانات إلى تدريب واختبار
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # تدريب النموذج
#     model.fit(X_train, y_train)

#     return model, X_train, X_test, y_train, y_test

# def evaluate_model(ml_task, model, X_test, y_test):
#     """
#     تقييم النموذج بناءً على نوع المهمة.
    
#     Args:
#         ml_task (str): نوع المهمة (classification, regression, clustering)
#         model: النموذج المدرب
#         X_test: بيانات الاختبار (المتغيرات المستقلة)
#         y_test: بيانات الاختبار (المتغير الهدف)
    
#     Returns:
#         tuple: (accuracy, chart_data)
#             - accuracy: دقة النموذج (أو None لو clustering)
#             - chart_data: بيانات المخطط (مثل confusion matrix للـ classification)
#     """
#     accuracy = None
#     chart_data = None

#     if ml_task in ['classification', 'regression']:
#         predictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, predictions) if ml_task == 'classification' else model.score(X_test, y_test)
#         if ml_task == 'classification':
#             cm = confusion_matrix(y_test, predictions)
#             chart_data = {
#                 'labels': [f"Class {i}" for i in range(len(cm))],
#                 'data': cm.tolist()
#             }
#     elif ml_task == 'clustering':
#         # للـ clustering، نرجع الـ cluster labels
#         chart_data = {
#             'labels': [f"Cluster {i}" for i in range(len(set(model.labels_)))],
#             'data': model.labels_.tolist()
#         }

#     return accuracy, chart_data

# @login_required
# def ml_model_result(request, analysis_id):
#     try:
#         # استرجاع التحليل والتأكد من صلاحية المستخدم
#         analysis = get_object_or_404(Analysis, id=analysis_id)
#         if analysis.prompt.user != request.user:
#             return HttpResponseForbidden("Not allowed BadGate")

#         # استرجاع البيانات المرتبطة
#         data_file = get_object_or_404(CleanedData, analysis=analysis)
#         ml_model_info = get_object_or_404(ML_Models, analysis=analysis)
#         hyperparams = get_object_or_404(ModelHyperParameters, ml_model=ml_model_info)

#         # استخراج معلومات النموذج والبيانات
#         ml_task = ml_model_info.model_type.lower()
#         ml_algo = ml_model_info.model
#         ml_hyperparams = hyperparams.hyperparams
#         scaller = data_file.scaller
#         normalier = data_file.normalier
#         encoder = data_file.encoder
#         data_path = data_file.clean_data.path

#         # جلب بيانات التدريب
#         data_provider = GetData(cleaned_data_path=data_path, cleaned_type=data_file.type, target=data_file.target_column)
#         try:
#             train_info = data_provider.get_train_data()
#             X, y = train_info['X'], train_info['y']
#         except Exception as e:
#             logger.error(f"Error in getting train data: {str(e)}")
#             return render(request, 'pages/ml-details.html', {'error': str(e)})

#         # تحويل الـ hyperparameters إلى dict إذا كانت JSON string
#         if isinstance(ml_hyperparams, str):
#             ml_hyperparams = json.loads(ml_hyperparams)

#         # تدريب النموذج
#         model, X_train, X_test, y_train, y_test = train_model(ml_task, ml_algo, ml_hyperparams, X, y)

#         # تقييم النموذج
#         accuracy, chart_data = evaluate_model(ml_task, model, X_test, y_test)

#         # حفظ أوزان النموذج
#         weights_path = os.path.join('weights', f'model_{analysis_id}.pkl')
#         os.makedirs(os.path.dirname(weights_path), exist_ok=True)
#         joblib.dump(model, weights_path)
#         ModerWheights.objects.create(model_name=ml_model_info, weights_file=weights_path)

#         # إعداد أسماء وأنواع الحقول للفورم
#         feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(X_train.shape[1])]
#         feature_types = [str(dtype) for dtype in X_train.dtypes] if isinstance(X_train, pd.DataFrame) else ['float' for _ in range(X_train.shape[1])]

#         prediction_result = None
#         input_values = None

#         if request.method == "POST":
#             form = PredictionForm(data=request.POST, feature_names=feature_names, feature_types=feature_types)
#             if form.is_valid():
#                 # عملية التوقع
#                 inputs = [form.cleaned_data[f] for f in feature_names]
#                 prediction_result = model.predict(np.array(inputs).reshape(1, -1))[0]
#         else:
#             form = PredictionForm(feature_names=feature_names, feature_types=feature_types)

#         # إعداد form_fields للـ template
#         form_fields = []
#         for field_name, field in form.fields.items():
#             field_type = "Numeric" if isinstance(X_train, pd.DataFrame) and pd.api.types.is_numeric_dtype(X_train[field_name]) else "Categorical"
#             form_fields.append({
#                 "field": form[field_name],
#                 "name": field_name,
#                 "type": field_type
#             })

#         context = {
#             'ml_task': ml_task,
#             'ml_algo': ml_algo,
#             'ml_hyperparams': json.dumps(ml_hyperparams, indent=2),
#             'data_path': data_path,
#             'scaller': scaller or "None",
#             'normalier': normalier or "None",
#             'encoder': encoder or "None",
#             'accuracy': round(accuracy * 100, 2) if accuracy else None,
#             'chart_data': json.dumps(chart_data) if chart_data else None,
#             'form': form,
#             'prediction_result': prediction_result,
#             'form_fields': form_fields,
#             'input_values': input_values,
#         }

#         return render(request, 'pages/ml-details.html', context)

#     except Exception as e:
#         logger.error(f"Error in ml_model_result: {str(e)}")
#         messages.error(request, f"Error loading results: {str(e)}")
#         return render(request, 'pages/ml-details.html', {'error': str(e)})


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden, HttpResponse
from django.contrib import messages
import logging
from .forms import PredictionForm, TrainingConfigForm
from sheet_models.models import Analysis, CleanedData, ML_Models, ModelHyperParameters
from sheet_models.data_collector import GetData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import json
import os
try:
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor = None, None, None, None

# إعداد الـ logger
logger = logging.getLogger(__name__)

# Dictionary لربط أسماء الخوارزميات بالكائنات
ALGORITHM_MAP = {
    'classification': {
        'Random Forest Classifier': RandomForestClassifier,
        'Logistic Regression': LogisticRegression,
        'Support Vector Classifier': SVC,
        'KNN Classifier': KNeighborsClassifier,
        'Decision Tree Classifier': DecisionTreeClassifier,
        'Gradient Boosting Classifier': GradientBoostingClassifier,
        'AdaBoost Classifier': AdaBoostClassifier,
        'Gaussian Naive Bayes': GaussianNB,
        'XGBoost': XGBClassifier if XGBClassifier else None,
        'LightGBM': LGBMClassifier if LGBMClassifier else None,
    },
    'regression': {
        'Random Forest Regressor': RandomForestRegressor,
        'Linear Regression': LinearRegression,
        'Support Vector Regressor': SVR,
        'KNN Regressor': KNeighborsRegressor,
        'Decision Tree Regressor': DecisionTreeRegressor,
        'Gradient Boosting Regressor': GradientBoostingRegressor,
        'AdaBoost Regressor': AdaBoostRegressor,
        'ElasticNet': ElasticNet,
        'XGBoost': XGBRegressor if XGBRegressor else None,
        'LightGBM': LGBMRegressor if LGBMRegressor else None,
    },
    'clustering': {
        'KMeans Clustering': KMeans,
        'DBSCAN': DBSCAN,
    }
}

def train_model(ml_task, ml_algo, ml_hyperparams, X, y):
    """
    تدريب النموذج بناءً على نوع المهمة والخوارزمية مع auto-fix للهايبر باراميترز الغلط.
    - لو عدد الصفوف > 1000 → ناخد عينة 25% فقط للتدريب + التست.
    """

    if ml_task not in ALGORITHM_MAP:
        raise ValueError(f"Unsupported ML task: {ml_task}")
    if ml_algo not in ALGORITHM_MAP[ml_task]:
        raise ValueError(f"Unsupported algorithm: {ml_algo} for task: {ml_task}")
    if ALGORITHM_MAP[ml_task][ml_algo] is None:
        raise ValueError(f"Algorithm {ml_algo} is not available. Please ensure required libraries are installed.")

    # --- اختيار الهايبر باراميترز ---
    selected_hyperparams = ml_hyperparams
    if isinstance(ml_hyperparams, list):
        for params in ml_hyperparams:
            if params is not None:
                selected_hyperparams = params
                break
        else:
            raise ValueError("No valid hyperparameters found in the list.")

    if not isinstance(selected_hyperparams, dict):
        raise ValueError(f"Expected hyperparameters to be a dict, got {type(selected_hyperparams)}")

    # --- تحويل القيم + auto-fix ---
    hypers = {}
    for key, value in selected_hyperparams.items():
        if isinstance(value, list):
            for v in value:
                if v is not None:
                    value = v
                    break
            else:
                print(f"[WARN] No valid value found for {key}, skipping.")
                continue

        # حاول نحول لرقم
        try:
            value = int(value)
        except (ValueError, TypeError):
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass

        # --- auto-fix rules ---
        if key in ["C", "alpha", "learning_rate"] and (not isinstance(value, (int, float)) or value <= 0):
            print(f"[WARN] {key}={value} is invalid, resetting to 1.0")
            value = 1.0
        elif key in ["n_estimators", "min_samples_split", "min_samples_leaf", "n_clusters"] and (
            not isinstance(value, int) or value <= 0
        ):
            print(f"[WARN] {key}={value} is invalid, resetting to 1")
            value = 1
        elif key == "max_depth" and (value is not None and (not isinstance(value, int) or value <= 0)):
            print(f"[WARN] max_depth={value} is invalid, resetting to None")
            value = None

        hypers[key] = value

    selected_hyperparams = hypers

    # --- إنشاء وتدريب النموذج ---
    model_class = ALGORITHM_MAP[ml_task][ml_algo]
    model = model_class(**selected_hyperparams)

    # --- اختيار عينة من البيانات لو حجمها كبير ---
    if len(X) > 1000:
        _, X_sample, _, y_sample = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        X, y = X_sample, y_sample
        print(f"[INFO] Using 25% sample of data → {len(X)} rows")

    # --- تقسيم Train/Test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def evaluate_model(ml_task, model, X_test, y_test):
    """
    تقييم النموذج بناءً على نوع المهمة.
    
    Args:
        ml_task (str): نوع المهمة (classification, regression, clustering)
        model: النموذج المدرب
        X_test: بيانات الاختبار (المتغيرات المستقلة)
        y_test: بيانات الاختبار (المتغير الهدف)
    
    Returns:
        tuple: (accuracy, chart_data)
            - accuracy: دقة النموذج (أو None لو clustering)
            - chart_data: بيانات المخطط (مثل confusion matrix للـ classification)
    """
    accuracy = None
    chart_data = None

    if ml_task in ['classification', 'regression']:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) if ml_task == 'classification' else model.score(X_test, y_test)
        if ml_task == 'classification':
            cm = confusion_matrix(y_test, predictions)
            chart_data = {
                'labels': [f"Class {i}" for i in range(len(cm))],
                'data': cm.tolist()
            }
    elif ml_task == 'clustering':
        # للـ clustering، نرجع الـ cluster labels
        chart_data = {
            'labels': [f"Cluster {i}" for i in range(len(set(model.labels_)))],
            'data': model.labels_.tolist()
        }

    return accuracy, chart_data

# @login_required
# def ml_model_result(request, analysis_id):
#     try:
#         # استرجاع التحليل والتأكد من صلاحية المستخدم
#         analysis = get_object_or_404(Analysis, id=analysis_id)
#         if analysis.prompt.user != request.user:
#             return HttpResponseForbidden("Not allowed BadGate")

#         # استرجاع البيانات المرتبطة
#         data_file = get_object_or_404(CleanedData, analysis=analysis)
#         ml_model_info = get_object_or_404(ML_Models, analysis=analysis)
#         hyperparams = get_object_or_404(ModelHyperParameters, ml_model=ml_model_info)

#         # استخراج معلومات النموذج والبيانات
#         ml_task = ml_model_info.model_type.lower()  # تحويل إلى lowercase للتأكد
#         ml_algo = ml_model_info.model
#         ml_hyperparams = hyperparams.hyperparams
#         scaller = data_file.scaller
#         normalier = data_file.normalier
#         encoder = data_file.encoder
#         data_path = data_file.clean_data.path

#         # جلب بيانات التدريب
#         data_provider = GetData(cleaned_data_path=data_path, cleaned_type=data_file.type, target=data_file.target_column)
#         try:
#             train_info = data_provider.get_train_data()
#             X, y = train_info['X'], train_info['y']
#         except Exception as e:
#             print("Error in getting train data:", str(e))
#             logger.error(f"Error in getting train data: {str(e)}")
#             return render(request, 'pages/ml-details.html', {'error': str(e)})

#         # تحويل الـ hyperparameters إلى dict إذا كانت JSON string
#         if isinstance(ml_hyperparams, str):
#             ml_hyperparams = json.loads(ml_hyperparams)

#         # تدريب النموذج
#         model, X_train, X_test, y_train, y_test = train_model(ml_task, ml_algo, ml_hyperparams, X, y)
        
#         # تقييم النموذج
#         accuracy, chart_data = evaluate_model(ml_task, model, X_test, y_test)

#         # حفظ أوزان النموذج
#         weights_path = os.path.join('weights', f'model_{analysis_id}.pkl')
#         os.makedirs(os.path.dirname(weights_path), exist_ok=True)
#         joblib.dump(model, weights_path)
#         ModerWheights.objects.create(model_name=ml_model_info, weights_file=weights_path)

#         # تحضير قائمة الأعمدة وأنواعها
#         feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(X_train.shape[1])]
#         feature_types = X_train.dtypes if isinstance(X_train, pd.DataFrame) else ['float' for _ in range(X_train.shape[1])]

#         # إنشاء الفورم
#         form = PredictionForm(request.POST or None)
#         from django import forms
#         # إضافة الحقول ديناميكيًا بعد إنشاء الفورم
#         for col, dtype in zip(feature_names, feature_types):
#             if hasattr(dtype, 'name'):
#                 dtype_str = dtype.name.lower()
#             else:
#                 dtype_str = str(dtype).lower()

#             if dtype_str in ['int64', 'float64', 'int32', 'float32', 'numeric']:
#                 field = forms.FloatField(label=col, required=True)
#             else:
#                 field = forms.CharField(label=col, required=True)

#             field.widget.attrs['data-type'] = dtype_str
#             form.fields[col] = field

#         # التعامل مع POST
#         prediction_result = None
#         input_values = None
#         if request.method == "POST" and form.is_valid():
#             inputs = [form.cleaned_data[f] for f in feature_names]
#             input_values = dict(zip(feature_names, inputs))
#             prediction_result = str(model.predict(np.array(inputs).reshape(1, -1))[0])


#         # إعداد form_fields للـ template
#         form_fields = []
#         for field_name, field in form.fields.items():
#             field_type = "Numeric" if isinstance(X_train, pd.DataFrame) and pd.api.types.is_numeric_dtype(X_train[field_name]) else "Categorical"
#             form_fields.append({
#                 "field": form[field_name],   # Field نفسه
#                 "name": field_name,          # اسم العمود
#                 "type": field_type           # النوع
#             })

#         # إعداد الـ context للـ template
#         context = {
#             'ml_task': ml_task,
#             'ml_algo': ml_algo,
#             'ml_hyperparams': json.dumps(ml_hyperparams, indent=2),
#             'data_path': data_path,
#             'scaller': scaller or "None",
#             'normalier': normalier or "None",
#             'encoder': encoder or "None",
#             'accuracy': round(accuracy * 100, 2) if accuracy else None,
#             'chart_data': json.dumps(chart_data) if chart_data else None,
#             'form': form,
#             'prediction_result': prediction_result,
#             'form_fields': form_fields,
#             'input_values': input_values,  # إضافة القيم المدخلة
#         }

#         return render(request, 'pages/ml-details.html', context)

#     except Exception as e:
#         print('=======================================================')
#         print("error", str(e))
#         print('=======================================================')
#         logger.error(f"Error in ml_model_result: {str(e)}")
#         messages.error(request, f"Error loading results: {str(e)}")
#         return render(request, 'pages/ml-details.html', {'error': str(e)})

@login_required
def ml_model_result(request, analysis_id):
    try:
        # جلب التحليل
        analysis = get_object_or_404(Analysis, id=analysis_id)
        if analysis.prompt.user != request.user:
            return HttpResponseForbidden("Not allowed BadGate")

        data_file = get_object_or_404(CleanedData, analysis=analysis)
        ml_model_info = get_object_or_404(ML_Models, analysis=analysis)
        hyperparams = get_object_or_404(ModelHyperParameters, ml_model=ml_model_info)

        ml_task = ml_model_info.model_type.lower()
        ml_algo = ml_model_info.model
        ml_hyperparams = hyperparams.hyperparams
        data_path = data_file.clean_data.path

        data_provider = GetData(
            cleaned_data_path=data_path,
            cleaned_type=data_file.type,
            target=data_file.target_column
        )

        try:
            train_info = data_provider.get_train_data()
            X, y = train_info['X'], train_info['y']
        except Exception as e:
            # ❌ هنا الخطأ → نرجع الفورم لليوزر يغير الإعدادات
            form = TrainingConfigForm(initial={
                "target_column": data_file.target_column,
                "ml_task": ml_task,
                "ml_algo": ml_algo,
                "hyperparams": json.dumps(ml_hyperparams, indent=2) if ml_hyperparams else "{}"
            })
            return render(request, 'pages/ml-details.html', {
                'error': str(e),
                'training_form': form,
            })

        # باقي الكود (تدريب وتقييم النموذج) ...
        model, X_train, X_test, y_train, y_test = train_model(
            ml_task, ml_algo, ml_hyperparams, X, y
        )
        accuracy, chart_data = evaluate_model(ml_task, model, X_test, y_test)

        # تجهيز السياق
        context = {
            'ml_task': ml_task,
            'ml_algo': ml_algo,
            'ml_hyperparams': json.dumps(ml_hyperparams, indent=2),
            'accuracy': round(accuracy * 100, 2) if accuracy else None,
            'chart_data': json.dumps(chart_data) if chart_data else None,
        }
        return render(request, 'pages/ml-details.html', context)

    except Exception as e:
        logger.error(f"Error in ml_model_result: {str(e)}")
        form = TrainingConfigForm()
        return render(request, 'pages/ml-details.html', {
            'error': str(e),
            'training_form': form,
        })




@login_required
def create_project(request, analysis_id):
    """
    Create new ML project with dataset upload and configuration
    Takes: POST data with project info, dataset, and configuration
    Returns: Rendered create_project.html with forms
    """
    try:
        analysis = get_object_or_404(Analysis, id=analysis_id)
        # Initialize forms
        project_form = MLProjectForm()
        dataset_form = DatasetUploadForm()
        config_form = None
        evaluation_form = None
        
        if request.method == 'POST':
            action = request.POST.get('action')
            
            if action == 'create_project':
                # Handle project creation
                project_form = MLProjectForm(request.POST)
                
                if project_form.is_valid():
                    project = project_form.save(commit=False)
                    project.analysis = analysis
                    project.user = request.user
                    project.save()
                    
                    log_system_event(project, 'INFO', 'New project created', 'create_project')
                    messages.success(request, 'Project created successfully!')
                    
                    # Create forms for next steps
                    config_form = MLConfigurationForm(pipeline_type=project.pipeline_type)
                    evaluation_form = EvaluationConfigForm(pipeline_type=project.pipeline_type)
                    
            elif action == 'upload_dataset':
                # Handle dataset upload
                project_id = request.POST.get('project_id')
                project = get_object_or_404(MLProject, id=project_id, user=request.user)
                
                dataset_form = DatasetUploadForm(request.POST, request.FILES)
                
                if dataset_form.is_valid():
                    # Save uploaded file
                    uploaded_file = request.FILES['dataset_file']
                    file_path = default_storage.save(
                        f'datasets/{project.id}/{uploaded_file.name}',
                        ContentFile(uploaded_file.read())
                    )
                    
                    # Analyze dataset
                    full_path = os.path.join(settings.MEDIA_ROOT, file_path)
                    dataset_analysis = analyze_dataset(full_path)
                    
                    # Create DatasetInfo
                    dataset_info = DatasetInfo.objects.create(
                        project=project,
                        file_path=full_path,
                        target_column=dataset_form.cleaned_data['target_column'],
                        **dataset_analysis
                    )
                    
                    log_system_event(project, 'INFO', 'Dataset uploaded and analyzed', 'create_project')
                    messages.success(request, 'Dataset uploaded successfully!')
                    
            elif action == 'configure_ml':
                # Handle ML configuration
                project_id = request.POST.get('project_id')
                project = get_object_or_404(MLProject, id=project_id, user=request.user)
                
                config_form = MLConfigurationForm(request.POST, pipeline_type=project.pipeline_type)
                
                if config_form.is_valid():
                    ml_config = config_form.save(commit=False)
                    ml_config.project = project
                    ml_config.save()
                    
                    log_system_event(project, 'INFO', 'ML configuration saved', 'create_project')
                    messages.success(request, 'Configuration saved successfully!')
                    
            elif action == 'start_training':
                # Start training pipeline
                project_id = request.POST.get('project_id')
                project = get_object_or_404(MLProject, id=project_id, user=request.user)
                
                result = start_training_pipeline(project)
                if result['success']:
                    messages.success(request, 'Training started successfully!')
                else:
                    messages.error(request, f"Training failed: {result['error']}")
        
        context = {
            'project_form': project_form,
            'dataset_form': dataset_form,
            'config_form': config_form,
            'evaluation_form': evaluation_form,
        }
        
        return render(request, 'pages/ml-details.html', context)
        
    except Exception as e:
        logger.error(f"Error in create_project: {str(e)}")
        messages.error(request, f"Error creating project: {str(e)}")
        return render(request, 'pages/ml-details.html', {'project_form': MLProjectForm()})

@login_required
def make_prediction(request, project_id):
    """
    Make prediction using trained model
    Takes: project_id and input feature values
    Returns: JSON response with prediction result
    """
    try:
        project = get_object_or_404(MLProject, id=project_id, user=request.user)
        
        # Get training result and check if model exists
        training_result = TrainingResult.objects.filter(
            project=project, status='completed'
        ).last()
        
        if not training_result or not training_result.model_path:
            return JsonResponse({
                'success': False,
                'error': 'No trained model available'
            })
        
        # Get dataset info for form validation
        dataset_info = DatasetInfo.objects.get(project=project)
        
        if request.method == 'POST':
            prediction_form = PredictionForm(request.POST, dataset_info=dataset_info)
            
            if prediction_form.is_valid():
                # Extract input data
                input_data = {}
                for field_name, value in prediction_form.cleaned_data.items():
                    input_data[field_name] = value
                
                # Make prediction
                prediction_result = make_model_prediction(training_result.model_path, input_data)
                
                # Save prediction request
                prediction_request = PredictionRequest.objects.create(
                    project=project,
                    input_data=input_data,
                    prediction_result=prediction_result['prediction'],
                    confidence_score=prediction_result.get('confidence')
                )
                
                log_system_event(project, 'INFO', 'Prediction made successfully', 'make_prediction')
                
                return JsonResponse({
                    'success': True,
                    'prediction': prediction_result['prediction'],
                    'confidence': prediction_result.get('confidence'),
                    'model_accuracy': training_result.metrics.get('accuracy', 'N/A')
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid input data',
                    'form_errors': prediction_form.errors
                })
        
        return JsonResponse({'success': False, 'error': 'Method not allowed'})
        
    except Exception as e:
        logger.error(f"Error in make_prediction: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@login_required
def project_list(request):
    """
    Display list of user's ML projects
    Takes: None
    Returns: Rendered project_list.html with projects
    """
    try:
        projects = MLProject.objects.filter(user=request.user).order_by('-created_at')
        
        # Get training status for each project
        project_data = []
        for project in projects:
            training_result = TrainingResult.objects.filter(project=project).last()
            project_data.append({
                'project': project,
                'training_result': training_result,
                'has_dataset': DatasetInfo.objects.filter(project=project).exists(),
                'has_config': MLConfiguration.objects.filter(project=project).exists(),
            })
        
        context = {
            'project_data': project_data
        }
        
        return render(request, 'project_list.html', context)
        
    except Exception as e:
        logger.error(f"Error in project_list: {str(e)}")
        messages.error(request, f"Error loading projects: {str(e)}")
        return render(request, 'project_list.html', {'project_data': []})

@login_required
def start_evaluation(request, project_id):
    """
    Start evaluation pipeline for trained model
    Takes: project_id
    Returns: JSON response with evaluation status
    """
    try:
        project = get_object_or_404(MLProject, id=project_id, user=request.user)
        
        # Check if training is completed
        training_result = TrainingResult.objects.filter(
            project=project, status='completed'
        ).last()
        
        if not training_result:
            return JsonResponse({
                'success': False,
                'error': 'No completed training found'
            })
        
        # Start evaluation
        result = start_evaluation_pipeline(project, training_result)
        
        if result['success']:
            log_system_event(project, 'INFO', 'Evaluation started', 'start_evaluation')
            return JsonResponse({
                'success': True,
                'message': 'Evaluation started successfully'
            })
        else:
            log_system_event(project, 'ERROR', f'Evaluation failed: {result["error"]}', 'start_evaluation')
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
        
    except Exception as e:
        logger.error(f"Error in start_evaluation: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@require_http_methods(["POST"])
def training_status_webhook(request, project_id):
    """
    Webhook to receive training status updates from training pipeline
    Takes: project_id and training status data
    Returns: JSON response confirming receipt
    """
    try:
        project = get_object_or_404(MLProject, id=project_id)
        data = json.loads(request.body)
        
        # Update training result
        training_result = TrainingResult.objects.filter(project=project).last()
        if training_result:
            training_result.status = data.get('status', training_result.status)
            training_result.metrics = data.get('metrics', training_result.metrics)
            training_result.model_path = data.get('model_path', training_result.model_path)
            training_result.training_time = data.get('training_time', training_result.training_time)
            training_result.error_message = data.get('error_message', training_result.error_message)
            
            if data.get('status') == 'completed':
                training_result.completed_at = datetime.now()
            
            training_result.save()
            
            log_system_event(project, 'INFO', f'Training status updated: {training_result.status}', 'training_status_webhook')
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        logger.error(f"Error in training_status_webhook: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt  
@require_http_methods(["POST"])
def evaluation_status_webhook(request, project_id):
    """
    Webhook to receive evaluation status updates from evaluation pipeline
    Takes: project_id and evaluation results data
    Returns: JSON response confirming receipt
    """
    try:
        project = get_object_or_404(MLProject, id=project_id)
        data = json.loads(request.body)
        
        # Get training result
        training_result = TrainingResult.objects.filter(project=project).last()
        if training_result:
            # Create or update evaluation result
            evaluation_result, created = EvaluationResult.objects.get_or_create(
                training_result=training_result,
                defaults={
                    'evaluation_metrics': data.get('metrics', {}),
                    'plots_data': data.get('plots', {}),
                    'confusion_matrix': data.get('confusion_matrix'),
                    'classification_report': data.get('classification_report'),
                    'feature_importance': data.get('feature_importance', {}),
                    'evaluation_time': data.get('evaluation_time')
                }
            )
            
            if not created:
                evaluation_result.evaluation_metrics = data.get('metrics', evaluation_result.evaluation_metrics)
                evaluation_result.plots_data = data.get('plots', evaluation_result.plots_data)
                evaluation_result.confusion_matrix = data.get('confusion_matrix', evaluation_result.confusion_matrix)
                evaluation_result.classification_report = data.get('classification_report', evaluation_result.classification_report)
                evaluation_result.feature_importance = data.get('feature_importance', evaluation_result.feature_importance)
                evaluation_result.evaluation_time = data.get('evaluation_time', evaluation_result.evaluation_time)
                evaluation_result.save()
            
            log_system_event(project, 'INFO', 'Evaluation results received', 'evaluation_status_webhook')
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        logger.error(f"Error in evaluation_status_webhook: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)})

# Utility functions

def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    Analyze uploaded dataset and return basic information
    Takes: file path to dataset
    Returns: Dictionary with dataset analysis
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        elif file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format")
        
        analysis = {
            'file_size': os.path.getsize(file_path),
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'has_missing_values': df.isnull().any().any()
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise

def start_training_pipeline(project: MLProject) -> Dict[str, Any]:
    """
    Start training pipeline for given project
    Takes: MLProject instance
    Returns: Dictionary with success status and results
    """
    try:
        # Get required data
        dataset_info = DatasetInfo.objects.get(project=project)
        ml_config = MLConfiguration.objects.get(project=project)
        
        # Create training result record
        training_result = TrainingResult.objects.create(
            project=project,
            status='pending',
            started_at=datetime.now()
        )
        
        # Initialize training pipeline
        training_pipeline = TrainingPipeline(
            project_id=project.id,
            dataset_path=dataset_info.file_path,
            config=ml_config,
            output_dir=f'models/{project.id}/'
        )
        
        # Start training (this should be async in production)
        result = training_pipeline.run()
        
        # Update training result
        training_result.status = 'completed' if result['success'] else 'failed'
        training_result.model_path = result.get('model_path')
        training_result.metrics = result.get('metrics', {})
        training_result.training_time = result.get('training_time')
        training_result.error_message = result.get('error') if not result['success'] else ''
        training_result.completed_at = datetime.now()
        training_result.save()
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting training pipeline: {str(e)}")
        return {'success': False, 'error': str(e)}

def start_evaluation_pipeline(project: MLProject, training_result: TrainingResult) -> Dict[str, Any]:
    """
    Start evaluation pipeline for trained model
    Takes: MLProject instance and TrainingResult
    Returns: Dictionary with success status and results
    """
    try:
        dataset_info = DatasetInfo.objects.get(project=project)
        
        # Initialize evaluation pipeline
        evaluation_pipeline = EvaluationPipeline(
            model_path=training_result.model_path,
            test_data_path=dataset_info.file_path,
            pipeline_type=project.pipeline_type,
            output_dir=f'evaluations/{project.id}/'
        )
        
        # Start evaluation
        result = evaluation_pipeline.run()
        
        if result['success']:
            # Create evaluation result
            EvaluationResult.objects.create(
                training_result=training_result,
                evaluation_metrics=result.get('metrics', {}),
                plots_data=result.get('plots', {}),
                confusion_matrix=result.get('confusion_matrix'),
                classification_report=result.get('classification_report'),
                feature_importance=result.get('feature_importance', {}),
                evaluation_time=result.get('evaluation_time')
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting evaluation pipeline: {str(e)}")
        return {'success': False, 'error': str(e)}

def make_model_prediction(model_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make prediction using trained model
    Takes: model file path and input data dictionary
    Returns: Dictionary with prediction results
    """
    try:
        import joblib
        
        # Load model
        model = joblib.load(model_path)
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            confidence = max(probabilities)
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

@login_required
def generate_config(request, project_id):
    """
    Generate ML configuration using FastAPI service
    Takes: project_id
    Returns: JSON response with generated configuration
    """
    try:
        from .communication import MLConfigurationService
        
        project = get_object_or_404(MLProject, id=project_id, user=request.user)
        dataset_info = DatasetInfo.objects.get(project=project)
        
        # Get user preferences from request
        user_preferences = {}
        if request.method == 'POST':
            user_preferences = json.loads(request.body).get('preferences', {})
        
        # Generate configuration
        config_service = MLConfigurationService()
        result = config_service.generate_and_apply_config(project, dataset_info, user_preferences)
        
        if result['success']:
            log_system_event(project, 'INFO', 'Configuration generated successfully', 'generate_config')
            return JsonResponse({
                'success': True,
                'message': 'Configuration generated successfully',
                'ml_config': {
                    'algorithm': result['ml_configuration'].algorithm,
                    'hyperparameters': result['ml_configuration'].hyperparameters,
                    'reasoning': result['suggestions_and_warnings']['reasoning']
                },
                'evaluation_config': result['evaluation_configuration'],
                'suggestions': result['suggestions_and_warnings']['suggestions'],
                'warnings': result['suggestions_and_warnings']['warnings']
            })
        else:
            log_system_event(project, 'ERROR', f'Configuration generation failed: {result["error"]}', 'generate_config')
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
    
    except Exception as e:
        logger.error(f"Error in generate_config: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@login_required
def config_service_status(request):
    """
    Check FastAPI configuration service status
    Takes: None
    Returns: JSON response with service status
    """
    try:
        from .communication import MLConfigurationService
        
        config_service = MLConfigurationService()
        status = config_service.validate_service_connection()
        
        return JsonResponse(status)
        
    except Exception as e:
        logger.error(f"Error checking config service status: {str(e)}")
        return JsonResponse({
            'connected': False,
            'error': str(e),
            'status': 'error'
        })
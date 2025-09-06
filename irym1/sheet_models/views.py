from django.shortcuts import render, redirect, get_object_or_404
from . import models
from django.conf import settings
from .forms import *
temps = settings.T_PATH
from smartsheet_sync.fast_apis_side import send_data_info, get_data_config
from sheet_models.data_collector import GetData
import json
from .data_cleaning_structure import CleaningConfig
from . import models as sm_models
from .data_science_unit import DataScienceEngine
import traceback
import logging
# Create your views here.
import os
import math
from sheet_models.data_cleaning_structure import (NULL_HANDLERS,
                                                  NORMALIZATION_METHODS,
                                                  LABELING_METHODS, 
                                                  OUTLIER_HANDLERS,
                                                  SCALERS,)
from pathlib import Path
from django.contrib.auth.decorators import login_required
from core_performance import performance_logger
from decimal import Decimal, InvalidOperation
from decimal import Decimal, InvalidOperation


logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}  # الملفات المسموحة
MAX_FILE_SIZE_MB = 10  # الحد الأقصى للحجم (10 MB)

@performance_logger
def handle_uploaded_file(f):
    """Extract and validate file extension safely."""
    ext = Path(f.name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    return ext

@performance_logger
def method_checker(method_name, method_dict, default=None):
    return method_name if method_name in method_dict.keys() else default

@performance_logger
def create_config(resp, data_provider):
    try:
        results = resp.get("results", {})

        # Boolean configs
        bools = results.get("bool_configs", {})
        remove_duplicates = bools.get("remove_duplicates", True)
        scaling = bools.get("scaling", True)
        normalization = bools.get("normalization", True)
        handling_outliers = bools.get("handling_outliers", True)
        feature_extraction = bools.get("feature_extraction", True)
        feature_selection = bools.get("feature_selection", True)

        # Methods with fallback and typo handling
        normalization_method = method_checker(
            results.get("normalization_method", {}).get("normalization_method")
            or results.get("normalization_method"),
            NORMALIZATION_METHODS,
            default="L2 Normalization"
        )

        scaling_method = method_checker(
            results.get("scaling_method", {}).get("scaling_method")
            or results.get("scaleing_method"),
            SCALERS,
            default="Standard Scaler"
        )

        global_nulls_handler = method_checker(
            results.get("nulls_method", {}).get("nulls_method")
            or results.get("null_method"),
            NULL_HANDLERS,
            default="Constant Imputation"
        )

        columns_outliers_handler = method_checker(
            results.get("outliers_method", {}).get("outliers_method")
            or results.get("outlier_method"),
            OUTLIER_HANDLERS,
            default="IQR Removal"
        )

        string_handler = method_checker(
            results.get("string_handler", "Label Encoder"),
            LABELING_METHODS,
            default="Label Encoder"
        )

        # ML task & model & hyperparams
        raw_task_type = results.get("task_type") or results.get("ml_task", {})
        ml_task_type = raw_task_type.get("task_type") if isinstance(raw_task_type, dict) else raw_task_type

        raw_model = results.get("model", {})
        ml_model_name = raw_model.get("model_name") if isinstance(raw_model, dict) else raw_model

        ml_trials = {}
        if results.get("hyperparams"):
            if isinstance(results["hyperparams"], list) and len(results["hyperparams"]) > 0:
                ml_trials = results["hyperparams"][0].get("hyperparameters", {})
            elif isinstance(results["hyperparams"], dict):
                ml_trials = results["hyperparams"].get("hyperparameters", {})

        # fallback if target_column missing
        target_col = results.get("target_column", {}).get("target_column")
        if not target_col:
            target_col = str(data_provider.get_loaded_data.columns[-1])

        config = CleaningConfig(
            pipeline_type="default_pipeline",
            remove_duplicates=remove_duplicates,
            scaling=scaling,
            normalization=normalization,
            handling_outliers=handling_outliers,
            feature_extraction=feature_extraction,
            feature_selection=feature_selection,

            normalization_method=normalization_method,
            scaling_method=scaling_method,
            feature_extraction_method=None,
            feature_selector_method=None,
            global_nulls_handler=global_nulls_handler,
            columns_nulls_handler=results.get("columns_nulls_handler", None),
            columns_outliers_handler=columns_outliers_handler,

            removal_columns=results.get("removal_columns", []),
            string_handler=string_handler,

            target_column=target_col,
            cleaned_data=None,

            # ML fields
            task_type=ml_task_type,
            model=ml_model_name,
            hyperparams=ml_trials,
        )

        return config

    except Exception as e:
        raise ValueError(f"something went wrong in config creation function -> {e}")


def save_ml_models(config, analysis_obj, logger):
    """
    Save ML models and hyperparameters from CleaningConfig.
    Only numeric values are stored (pick first if list).
    """
    try:
        if not config.model:
            logger.warning("⚠️ No ML model specified in config, skipping save.")
            return

        # Create ML model record
        ml_model_obj = sm_models.ML_Models.objects.create(
            analysis=analysis_obj,
            model=config.model,
            model_type=config.task_type,
            description=f"Auto-generated config for {config.model}"
        )

        # Normalize hyperparams list
        # Normalize: دايمًا dict → JSONField في الداتا بيز
        hyperparams_json = config.hyperparams or {}

        sm_models.ModelHyperParameters.objects.create(
            ml_model=ml_model_obj,
            hyperparams=hyperparams_json
        )

        logger.info(f"✅ Saved ML model '{config.model}' with task type '{config.task_type}'")

    except Exception as e:
        logger.error(f"❌ Failed to save ML models/hyperparameters: {e}")

@login_required
@performance_logger
def prompt_room(request):
    prompt_form = PromptForm(request.POST or None)

    if request.method == "POST":
        data_file = request.FILES.get("upload")

        try:
            # === Validate form and file ===
            if not prompt_form.is_valid():
                return render(request, temps["prompt-room"], {
                    "prompt_form": prompt_form,
                    "error_message": "Invalid prompt form."
                })

            if not data_file:
                return render(request, temps["prompt-room"], {
                    "prompt_form": prompt_form,
                    "error_message": "No file uploaded."
                })

            if data_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return render(request, temps["prompt-room"], {
                    "prompt_form": prompt_form,
                    "error_message": f"File too large. Max size is {MAX_FILE_SIZE_MB}MB."
                })

            # === Validate extension ===
            try:
                file_ext = handle_uploaded_file(data_file)
            except ValueError as e:
                logger.warning(f"File validation error: {e}")
                return render(request, temps["prompt-room"], {
                    "prompt_form": prompt_form,
                    "error_message": str(e)
                })

            # === Save prompt ===
            prompt = prompt_form.cleaned_data["prompt"]
            prompt_obj = models.Prompt.objects.create(user=request.user, prompt=prompt)

            # === Save raw data ===
            data_size_kb = round(data_file.size / 1024, 2)
            raw_data = models.RawData.objects.create(
                prompt=prompt_obj,
                data=data_file,
                size=data_size_kb
            )

            # === Load dataset ===
            data_file.seek(0)
            try:
                data_provider = GetData(data_file, file_ext)
            except Exception as e:
                logger.error(f"Dataset load failed: {e}")
                return render(request, temps["prompt-room"], {
                    "prompt_form": prompt_form,
                    "error_message": "Could not read dataset. Please upload a valid CSV/Excel/JSON file."
                })

            # === Send dataset to FastAPI ===
            resp1 = send_data_info(data_provider, prompt=prompt)
            resp2 = get_data_config(resp1.get("data_id", ""))
            print("===============================================================")
            print(resp2)
            print("===============================================================")
            logger.debug("FastAPI response: %s", resp2)

            # === Build Cleaning Config ===
            try:
                config = create_config(resp2, data_provider)
            except Exception as e:
                logger.error(f"Config creation failed: {e}")
                return render(request, temps["prompt-room"], {
                    "prompt_form": prompt_form,
                    "error_message": "Failed to create data cleaning config. Please try again."
                })

            # === Save Analysis ===
            results = resp2.get("results", {})
            summary = results.get("summary_result", {}) or {}
            analysis_obj = sm_models.Analysis.objects.create(
                prompt=prompt_obj,
                title=summary.get("data_title", "Untitled Dataset"),
                description=summary.get("data_description", "No description available.")
            )
            save_ml_models(config, analysis_obj, logger)

            # # === Save ML Models + Hyperparameters ===
            # try:
            #     ml_experiment = results.get("ml_experiment", {})  # FastAPI returns MLExperimentSchema
            #     if ml_experiment:
            #         task_type = ml_experiment.get("task_type")
            #         for model_cfg in ml_experiment.get("models", []):
            #             ml_model_obj = sm_models.ML_Models.objects.create(
            #                 analysis=analysis_obj,
            #                 model=model_cfg.get("model_name"),
            #                 model_type=task_type,
            #                 description=f"Auto-generated config for {model_cfg.get('model_name')}"
            #             )
            #             # Save hyperparams safely
            #             for trial in model_cfg.get("trials", []):
            #                 for param_name, value in trial.get("hyperparameters", {}).items():
            #                     # Convert to safe Decimal
            #                     safe_value = Decimal(0)
            #                     try:
            #                         if isinstance(value, (int, float)):
            #                             if not (math.isnan(value) or math.isinf(value)):
            #                                 safe_value = Decimal(str(value))
            #                     except (InvalidOperation, TypeError):
            #                         safe_value = Decimal(0)

            #                     sm_models.ModelHyperParameters.objects.create(
            #                         ml_model=ml_model_obj,
            #                         parameter_name=param_name,
            #                         value=safe_value
            #                     )
            # except Exception as e:
                
            #     logger.error(f"Failed to save ML models/hyperparameters: {e}") 
        # === Run Engine ===
            engine = DataScienceEngine(
                analysis_obj=analysis_obj,
                cleaning_configs=config,
                dataset=data_provider.get_loaded_data,
                cols_description=results.get("columns_description", []),
                target=config.target_column,
                scaller=config.scaling_method,
                normalier=config.normalization_method, 
                encoder=config.string_handler,
            )
            eng_result = engine.start()
            logger.info(f"Engine result: {eng_result}")

            return render(request, temps["summary-result"], {
                "dash_id": analysis_obj.id,
                "title": summary.get("data_title"),
                "data_title": summary.get("data_title"),
                "description": summary.get("data_description")
            })

        except Exception as e:
            logger.exception(f"Error in prompt_room: {e}")
            return render(request, temps["prompt-room"], {
                "prompt_form": prompt_form,
                "danger_message": "Your data has something wrong. Please check that the data is in English, not empty, and in JSON/CSV/Excel format.",
                "error_message": f"Server error occurred: {str(e)}"
            })

    return render(request, temps["prompt-room"], {"prompt_form": prompt_form})


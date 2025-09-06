import numpy as np
import pandas as pd
import os
import matplotlib
from typing import Optional, Dict, List, Any
import json
from pydantic import BaseModel, Field
from dashboard import models as ds_models
from .models import Analysis, CleanedData
import datetime

from io import BytesIO
from django.core.files.base import ContentFile

from .data_cleaning_structure import (DataCleaningPipelineConfig,
                                     DataVisualizationPipelineConfig,
                                     BuildingCleaningPipeline,
                                     CleaningConfig, 
                                     )

from .data_training_structure import DataTrainingPipelineConfig, MODELS
from .data_evaluation_structure import (run_evaluation_pipeline,
                                        ModelComparator,
                                        load_and_evaluate_model,
                                        EvaluationConfig)

from core_performance import performance_logger
import math
"""
class PipelineManager:
    def __init__(self):
        self.cleaner_config = DataCleaningPipelineConfig()
        self.trainer_config = DataTrainingPipelineConfig()
        # self.tester_config = DataEvaluationPipelineConfig()
        self.visualizer_config = DataVisualizationPipelineConfig()

    def json_handler(self, json_config):
        if isinstance(json_config, str):
            return json.loads(json_config)
        elif isinstance(json_config, dict):
            return json_config
        else:
            raise TypeError("Config must be a dict or JSON string")

    def set_cleaner_config(self, config_json):
        json_config = self.json_handler(config_json)
        self.cleaner_config = DataCleaningPipelineConfig(
            pipeline_type=json_config.get("type", ""),
            remove_duplicates=json_config.get("remove_duplicate", True),
            columns_nulls_handler={k: v for d in json_config.get("columns_nulls_handler", []) for k, v in d.items()},
            columns_outliers_handler={k: v for d in json_config.get("columns_outliers_handler", []) for k, v in d.items()},
            removal_columns=json_config.get("removal_columns", []),
            normalization_method=json_config.get("normalization_method"),
            cleaned_data=json_config.get("cleaned_data"),
            string_columns_handler=json_config.get("string_handler")
        )

    def set_trainer_config(self, config_json):
        json_config = self.json_handler(config_json)
        self.trainer_config = DataTrainingPipelineConfig(
            pipeline_type=json_config.get("type", ""),
            algorithm=json_config.get("algorithm", ""),
            hyperparameters=json_config.get("hyperparameters", {}),
            cross_validation_folds=json_config.get("cross_validation_folds", 5),
            training_data=json_config.get("training_data"),
            validation_split=json_config.get("validation_split", 0.2),
            random_seed=json_config.get("random_seed"),
            trained_model=json_config.get("trained_model")
        )

    def set_tester_config(self, config_json):
        json_config = self.json_handler(config_json)
        self.tester_config = DataEvaluationPipelineConfig(
            pipeline_type=json_config.get("type", ""),
            evaluation_metrics=json_config.get("evaluation_metrics", []),
            test_data=json_config.get("test_data"),
            batch_size=json_config.get("batch_size"),
            save_results=json_config.get("save_results", False),
            results_format=json_config.get("results_format", "json"),
            evaluation_results=json_config.get("evaluation_results")
        )

    def set_visualizer_config(self, config_json):
        json_config = self.json_handler(config_json)
        self.visualizer_config = DataVisualizationPipelineConfig(
            pipeline_type=json_config.get("type", ""),
            chart_types=json_config.get("chart_types", {}),
            color_palette=json_config.get("color_palette"),
            show_grid=json_config.get("show_grid", True),
            figure_size=tuple(json_config.get("figure_size", (8, 6))),
            save_plots=json_config.get("save_plots", False),
            output_format=json_config.get("output_format", "png"),
            generated_plots=json_config.get("generated_plots")
        )
    
    def get_config(self, config_name: str):

        mapping = {
            "cleaner": self.cleaner_config,
            "visualizer": self.visualizer_config,
            "trainer": self.trainer_config,
            "tester": self.tester_config
        }
        try:
            return mapping[config_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown config name: {config_name}. Allowed: {list(mapping.keys())}")

    @property
    def get_cleaner_config(self):
        return self.cleaner_config
    
    @property
    def get_visualizer_config(self):
        return self.visualizer_config
    
    @property
    def get_trainer_config(self):
        return self.trainer_config
    
    @property
    def get_tester_config(self):
        return self.tester_config
"""

# =====================
# Result Models

class SummaryResultModel(BaseModel):
    title: str = Field(...), 
    description: str = Field(...)
    
class StatisticalResultModel(BaseModel):
    description: Optional[str] = None
    stats_table: Optional[Dict] = None  # مثلاً mean, std, min, max
    correlations: Optional[Dict] = None

class ML_ResultModel(BaseModel):
    model_name: Optional[str] = None
    metrics: Optional[Dict] = None
    feature_importance: Optional[Dict] = None

class VisualizationResultModel(BaseModel):
    charts: Optional[Dict] = None  # مثلا {"hist_age": "path/to/file.png"}

class ResultModel(BaseModel):
    summary_result: Optional[SummaryResultModel] = Field(None, description="Summary of the dataset")
    statistical_result: Optional[StatisticalResultModel] = Field(None, description="Statistical analysis result")
    ml_result: Optional[ML_ResultModel] = Field(None, description="ML training and evaluation result")
    vis_result: Optional[VisualizationResultModel] = Field(None, description="Visualization result")

# =====================
# DataScience Engine

def make_json_serializable(obj):
    """
    Convert objects into JSON-serializable format.
    Handles numpy types, pandas types, datetime, sets, NaN, inf, -inf.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        # تحويل NaN أو inf إلى None
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple, set, pd.Series)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, datetime.timedelta)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if obj is None or obj == float('nan'):
        return None
    return obj

class DataScienceEngine:
    def __init__(self, *args, **kwargs):
        self.all_configs: Optional[Dict]      = kwargs.get("all_configs")
        self.cleaning_configs: CleaningConfig = kwargs.get("cleaning_configs")
        self.vis_configs: Optional[Dict]      = kwargs.get("vis_configs")
        self.ml_configs: Optional[Dict]       = kwargs.get("ml_configs")
        self.eval_configs: Optional[Dict]     = kwargs.get("eval_configs")
        self.analysis_obj: Optional[Analysis] = kwargs.get("analysis_obj")
        self.dataset: Optional[pd.DataFrame]  = kwargs.get("dataset")
        self.cleaned_data: Optional[pd.DataFrame]  = kwargs.get("cleaned_data")
        self.target : Optional[str] = kwargs.get("target")
        self.cols_description = kwargs.get("cols_description")
        self.scaller = kwargs.get("scaller")
        self.normalier = kwargs.get("normalier")
        self.encoder = kwargs.get("encoder")
        
        # تنظيف الداتا
        self.cleaner = BuildingCleaningPipeline(config=self.cleaning_configs, raw_data=self.dataset)

        # باقي المكونات placeholders
        self.trainer = None
        self.evaluator = None
        self.visualizer = None

        # النتائج
        self.results = ResultModel()
        
    @performance_logger
    def start(self):
        # 1- Clean data
        self.cleaner.clean()
        clean_data = self.cleaner.cleaning_data  # attribute مش function
        # 2- Statistical analysis
        self.results.statistical_result = self._generate_statistics(clean_data,
                                                                    raw_data=self.cleaner.data,
                                                                    cols_descriptions=self.cols_description,
                                                                    cleaned_data_commbained=clean_data,
                                                                    scaller=self.scaller,
                                                                    normalier=self.normalier,
                                                                    encoder=self.encoder,
                                                                    )

        # 3- Summary
        # self.results.summary_result = self._generate_summary(clean_data)

        # 4- Visualization
        # self.results.vis_result = self._generate_visualizations(clean_data)

        # 5- ML training
        # self.results.ml_result = self._train_and_evaluate(clean_data)

        return self.results

    


    @performance_logger
    def _generate_statistics(
        self, 
        clean_data: pd.DataFrame, 
        raw_data: pd.DataFrame, 
        cols_descriptions: list,
        cleaned_data_commbained: pd.DataFrame = None,
        scaller= None,
        normalier=None,
        encoder=None,
        extension: str = "csv"   # الافتراضي CSV
    ) -> StatisticalResultModel:

        # --- Descriptive Statistics ---
        desc = clean_data.describe(include="all").rename(index={
            "25%": "q25",
            "50%": "median",
            "75%": "q75",
            "mean": "Mean",
            "std": "StdDev",
            "min": "Min",
            "max": "Max",
            "count": "Count"
        }).to_dict()
        desc = make_json_serializable(desc)

        # --- Correlations ---
        corr = clean_data.corr(numeric_only=True).to_dict()
        corr = make_json_serializable(corr)

        # --- Data Quality ---
        dq = pd.DataFrame({
            "column": raw_data.columns,
            "nulls": [int(raw_data[col].isnull().sum()) for col in raw_data.columns],
            "duplicates": [int(raw_data[col].duplicated().sum()) for col in raw_data.columns],
        }).to_dict(orient="records")
        dq = make_json_serializable(dq)

        # --- Ensure cols_descriptions is list ---
        if not cols_descriptions:
            cols_descriptions = []
        cols_descriptions_dict = {dic["column"]: dic["description"] for dic in cols_descriptions}
        cols_descriptions_dict = make_json_serializable(cols_descriptions_dict)

        # --- Save to DB ---
        ds_models.DescriptiveStatistics.objects.update_or_create(
            analysis=self.analysis_obj,
            defaults={"table": desc}
        )
        ds_models.Correlation.objects.update_or_create(
            analysis=self.analysis_obj,
            defaults={"table": corr}
        )
        ds_models.QualityReport.objects.update_or_create(
            analysis=self.analysis_obj,
            defaults={"table": dq}
        )
        ds_models.ColumnDescription.objects.update_or_create(
            analysis=self.analysis_obj,
            defaults={"table": cols_descriptions_dict}  # directly save
        )

        # --- Save Cleaned Data (combined) to DB ---
        if cleaned_data_commbained is not None:
            buffer = BytesIO()
            filename = f"cleaned_data.{extension}"

            if extension == "csv":
                cleaned_data_commbained.to_csv(buffer, index=False)
            elif extension == "excel":
                cleaned_data_commbained.to_excel(buffer, index=False)
            elif extension == "json":
                cleaned_data_commbained.to_json(buffer, orient="records")
            elif extension == "xml":
                cleaned_data_commbained.to_xml(buffer, index=False)
            else:
                raise ValueError(f"Unsupported file type: {extension}")

            buffer.seek(0)

            target_column = self.target or cleaned_data_commbained.columns[-1]

            cleaned_file, _ = CleanedData.objects.update_or_create(
                analysis=self.analysis_obj,
                defaults={
                    "target_column": target_column,
                    "size": buffer.getbuffer().nbytes,
                    "type": extension,
                    "scaller": scaller if scaller else None,
                    "normalier": normalier if normalier else None,
                    "encoder": encoder if encoder else None,
                }, 
            )
            cleaned_file.clean_data.save(filename, ContentFile(buffer.read()))
            buffer.close()

        # --- Return Pydantic Model ---
        return StatisticalResultModel(
            description="Basic statistical summary",
            stats_table=desc,
            correlations=corr,
            data_quality=dq,
            columns_description=cols_descriptions
        )

    
    @performance_logger
    def _generate_summary(self, df: pd.DataFrame) -> SummaryResultModel:
        text = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
        key_points = {
            "num_rows": df.shape[0],
            "num_cols": df.shape[1],
            "columns": list(df.columns)
        }
        return SummaryResultModel(text_summary=text, key_points=key_points)

    @performance_logger
    def _generate_visualizations(self, df: pd.DataFrame) -> VisualizationResultModel:
        # placeholder → ممكن تضيف matplotlib/seaborn هنا
        charts = {"example": "path/to/example_chart.png"}
        return VisualizationResultModel(charts=charts)

    @performance_logger
    def _train_and_evaluate(self, df: pd.DataFrame) -> ML_ResultModel:
        # placeholder → هنا هتحط ML pipeline
        metrics = {"accuracy": 0.85, "f1_score": 0.82}
        return ML_ResultModel(model_name="RandomForest", metrics=metrics)
    
    
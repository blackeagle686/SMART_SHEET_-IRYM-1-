import numpy as np
import pandas as pd
import os
import matplotlib
from typing import Optional, Dict, List, Any
import json
import plotly.express as px

from typing import Dict, List, Optional, Any, Tuple, Literal
from pydantic import BaseModel, Field
from django.shortcuts import get_object_or_404
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from scipy import stats

# 1. Encoding (تحويل الفئات الرقمية)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# 2. Feature Extraction & Dimensionality Reduction
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.feature_selection import SelectKBest, chi2  # Feature selection methods

# 3. Normalization & Scaling
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    Normalizer
)

# 4. Imputation (تعويض القيم الناقصة)
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest, f_classif, f_regression,
    RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dashboard.subprocess import performance_logger

from io import BytesIO

from django.core.files.base import ContentFile
from .models import CleanedData, Analysis


FEATURE_SELECTORS = {
    # بسيط – يشيل الـ features اللي الـ variance بتاعها قليل جدًا
    "Variance Threshold": lambda: VarianceThreshold(threshold=0.0),

    # لاختيار K features الأحسن بناءً على ANOVA F-test (تصنيف)
    "Select K Best (ANOVA)": lambda: SelectKBest(score_func=f_classif, k=10),

    # لاختيار K features الأحسن بناءً على F-test (انحدار)
    "Select K Best (Regression)": lambda: SelectKBest(score_func=f_regression, k=10),

    # Recursive Feature Elimination باستخدام Logistic Regression
    "RFE (Logistic Regression)": lambda: RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10),

    # Recursive Feature Elimination باستخدام Random Forest
    "RFE (Random Forest)": lambda: RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=10)
}


SCALERS = {
    "Standard Scaler": StandardScaler,
    "Min-Max Scaler": MinMaxScaler,
    "Max-Abs Scaler": MaxAbsScaler,
    "Robust Scaler": RobustScaler,
    "Power Transformer (Yeo-Johnson)": lambda: PowerTransformer(method="yeo-johnson"),
    "Power Transformer (Box-Cox)": lambda: PowerTransformer(method="box-cox"),  # only positive data
    "Quantile Transformer (Uniform)": lambda: QuantileTransformer(output_distribution="uniform"),
    "Quantile Transformer (Normal)": lambda: QuantileTransformer(output_distribution="normal"),
    "L2 Normalizer": Normalizer
}

"""
=================> SCALERS use case <=====================

from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize features (mean=0, std=1)
scaler = SCALERS["Standard Scaler"]()
X_scaled = scaler.fit_transform(X)

# Scale features to [0, 1] range
minmax_scaler = SCALERS["Min-Max Scaler"]()
X_minmax = minmax_scaler.fit_transform(X)

# Robust scaling (good with outliers)
robust_scaler = SCALERS["Robust Scaler"]()
X_robust = robust_scaler.fit_transform(X)

# Power transform to make data more Gaussian-like (Yeo-Johnson)
pt = SCALERS["Power Transformer (Yeo-Johnson)"]()
X_power = pt.fit_transform(X)

# Normalize samples to unit L2 norm
normalizer = SCALERS["L2 Normalizer"]()
X_normalized = normalizer.fit_transform(X)

"""

LABELING_METHODS = {
    "Label Encoder": LabelEncoder,
    "One-Hot Encoder": lambda: OneHotEncoder(sparse=False),  # returns instance
    "Ordinal Encoder": OrdinalEncoder,
    "Count Vectorizer": CountVectorizer,          # text → counts
    "TF-IDF Vectorizer": TfidfVectorizer          # text → weighted counts
}

"""
=================> LABELING_METHODS use case <=====================

import pandas as pd

# Example categorical labels
labels = ["cat", "dog", "dog", "cat", "mouse"]

# Label encoding (convert categories to integers)
label_encoder = LABELING_METHODS["Label Encoder"]()
encoded_labels = label_encoder.fit_transform(labels)

# One-hot encoding categorical features (pandas example)
df = pd.DataFrame({"color": ["red", "blue", "green", "blue"]})
one_hot_encoder = LABELING_METHODS["One-Hot Encoder"](sparse=False)
encoded_features = one_hot_encoder.fit_transform(df)

# Ordinal encoding (ordered categories)
ordinal_encoder = LABELING_METHODS["Ordinal Encoder"]()
ordinal_encoded = ordinal_encoder.fit_transform(df)

# Text feature extraction with Count Vectorizer
texts = ["I love AI", "AI is great"]
count_vec = LABELING_METHODS["Count Vectorizer"]()
X_counts = count_vec.fit_transform(texts)

# Text feature extraction with TF-IDF Vectorizer
tfidf_vec = LABELING_METHODS["TF-IDF Vectorizer"]()
X_tfidf = tfidf_vec.fit_transform(texts)

"""

NORMALIZATION_METHODS = {
    "L1 Normalization": lambda: Normalizer(norm="l1"),
    "L2 Normalization": lambda: Normalizer(norm="l2"),
    "Max Normalization": lambda: Normalizer(norm="max"),
    "Min-Max Scaling (0-1)": MinMaxScaler,
    "Max-Abs Scaling": MaxAbsScaler
}

"""
=================> NORMALIZATION_METHODS use case <=====================

import numpy as np
import pandas as pd

data = {
    "feature1": [1, 2, 3, 4],
    "feature2": [2, 3, 4, 5]
}
df = pd.DataFrame(data)

# L1 Normalization: each sample scaled so sum of absolute values = 1
l1_norm = NORMALIZATION_METHODS["L1 Normalization"]()
X_l1 = l1_norm.fit_transform(df)

# L2 Normalization: each sample scaled so Euclidean norm = 1
l2_norm = NORMALIZATION_METHODS["L2 Normalization"]()
X_l2 = l2_norm.fit_transform(df)

# Max Normalization: each sample scaled by its max absolute value
max_norm = NORMALIZATION_METHODS["Max Normalization"]()
X_max = max_norm.fit_transform(df)

# Min-Max Scaling (0-1)
minmax_scaler = NORMALIZATION_METHODS["Min-Max Scaling (0-1)"]()
X_minmax = minmax_scaler.fit_transform(df)

# Max-Abs Scaling (good for sparse data)
maxabs_scaler = NORMALIZATION_METHODS["Max-Abs Scaling"]()
X_maxabs = maxabs_scaler.fit_transform(df)

"""

NULL_HANDLERS = {
    "Drop Rows": lambda X: X.dropna(),  # Drops any row with nulls; expects a pandas DataFrame
    "Drop Columns": lambda X: X.dropna(axis=1),  # Drops columns with nulls
    "Mean Imputation": lambda: SimpleImputer(strategy="mean"),
    "Median Imputation": lambda: SimpleImputer(strategy="median"),
    "Most Frequent Imputation": lambda: SimpleImputer(strategy="most_frequent"),
    "Constant Imputation": lambda: SimpleImputer(strategy="constant", fill_value=0),
    "KNN Imputation": lambda: KNNImputer(n_neighbors=5),
    "Iterative Imputation": lambda: IterativeImputer(max_iter=10, random_state=0)
}

"""
=================> NULL_HANDLERS use case <=====================

# Sample DataFrame with nulls
data = {
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, np.nan, np.nan, 4]
}
df = pd.DataFrame(data)

# Drop rows with any nulls
df_dropped_rows = NULL_HANDLERS["Drop Rows"](df)

# Drop columns with any nulls
df_dropped_cols = NULL_HANDLERS["Drop Columns"](df)

imputer = NULL_HANDLERS["Mean Imputation"]()  # instantiate the transformer

# Fit on data and transform
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

knn_imputer = NULL_HANDLERS["KNN Imputation"]()
df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# Iterative Imputer
iter_imputer = NULL_HANDLERS["Iterative Imputation"]()
df_iter_imputed = pd.DataFrame(iter_imputer.fit_transform(df), columns=df.columns)

"""

OUTLIER_HANDLERS = {
    "Z-Score Removal": lambda df, thresh=3: df[
        (np.abs(stats.zscore(df.select_dtypes(include=[np.number]), nan_policy='omit')) < thresh).all(axis=1)
    ],
    "IQR Removal": lambda df: df[
        ~(
            (df.select_dtypes(include=[np.number]) < (df.quantile(0.25) - 1.5 * (df.quantile(0.75) - df.quantile(0.25)))) |
            (df.select_dtypes(include=[np.number]) > (df.quantile(0.75) + 1.5 * (df.quantile(0.75) - df.quantile(0.25))))
        ).any(axis=1)
    ],
    "Isolation Forest": lambda df: df[
        IsolationForest(contamination=0.05, random_state=0)
        .fit_predict(df.select_dtypes(include=[np.number])) == 1
    ],
    "Elliptic Envelope": lambda df: df[
        EllipticEnvelope(contamination=0.05)
        .fit_predict(df.select_dtypes(include=[np.number])) == 1
    ]
}

"""
=================> OUTLIER_HANDLERS use case <=====================

# Sample DataFrame with outliers
data = {
    'A': [10, 12, 14, 15, 1000],  # 1000 is an outlier
    'B': [20, 22, 23, 24, -999],  # -999 is an outlier
    'C': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Remove outliers using Z-Score method
df_zscore = OUTLIER_HANDLERS["Z-Score Removal"](df, thresh=3)

# Remove outliers using IQR method
df_iqr = OUTLIER_HANDLERS["IQR Removal"](df)

# Remove outliers using Isolation Forest
df_isoforest = OUTLIER_HANDLERS["Isolation Forest"](df)

# Remove outliers using Elliptic Envelope
df_elliptic = OUTLIER_HANDLERS["Elliptic Envelope"](df)

"""

"""
CHART_CONFIG = {
    "Histogram": {
        "required": ["x"],
        "optional": ["color", "nbins", "barmode"]
    },
    "KDE Plot": {
        "required": ["x"],
        "optional": ["color"]
    },
    "Box_plot": {
        "required": ["x", "y"],
        "optional": ["color", "points"]  # points="all" to show all observations
    },

    "Scatter_plot": {
        "required": ["x", "y"],
        "optional": ["color", "size", "hover_name"]
    },
    "Pair_plot": {
        "required": ["dimensions"],  # list of column names
        "optional": ["color"]
    },

    "Bar_plot": {
        "required": ["x", "y"],
        "optional": ["color", "barmode"]
    },
    "Count_plot": {
        "required": ["x"],
        "optional": ["color"]
    },

    "Line_plot": {
        "required": ["x", "y"],
        "optional": ["color", "line_group"]
    },
    "Area_plot": {
        "required": ["x", "y"],
        "optional": ["color", "groupnorm"]
    },

    "Radar_plot": {
        "required": ["r", "theta"],  # r = values, theta = categories
        "optional": ["color"]
    },
    "Treemap_plot": {
        "required": ["path", "values"],  # path can be list of hierarchical columns
        "optional": ["color"]
    },
    "Pie_plot": {
        "required": ["names", "values"],
        "optional": ["color", "hole"]  # hole for donut charts
    }
}

"""

CHARTS = {
    # Distribution & Frequency
    "Histogram": lambda df, x=None, color=None, **kwargs: px.histogram(df, x=x, color=color, **kwargs),
    "KDE Plot": lambda df, x=None, color=None, **kwargs: px.histogram(df, x=x, color=color, marginal="violin", histnorm="probability density", **kwargs),
    "Box_plot": lambda df, x=None, y=None, color=None, **kwargs: px.box(df, x=x, y=y, color=color, **kwargs),

    # Relationships
    "Scatter_plot": lambda df, x=None, y=None, color=None, size=None, **kwargs: px.scatter(df, x=x, y=y, color=color, size=size, **kwargs),
    "Pair_plot": lambda df, dimensions=None, color=None, **kwargs: px.scatter_matrix(df, dimensions=dimensions, color=color, **kwargs),

    # Categorical Comparisons
    "Bar_plot": lambda df, x=None, y=None, color=None, **kwargs: px.bar(df, x=x, y=y, color=color, **kwargs),
    "Count_plot": lambda df, x=None, color=None, **kwargs: px.histogram(df, x=x, color=color, **kwargs),

    # Time Series
    "Line_plot": lambda df, x=None, y=None, color=None, **kwargs: px.line(df, x=x, y=y, color=color, **kwargs),
    "Area_plot": lambda df, x=None, y=None, color=None, **kwargs: px.area(df, x=x, y=y, color=color, **kwargs),

    # Advanced
    "Radar_plot": lambda df, r=None, theta=None, color=None, **kwargs: px.line_polar(df, r=r, theta=theta, color=color, line_close=True, **kwargs),
    "Treemap_plot": lambda df, path=None, values=None, color=None, **kwargs: px.treemap(df, path=path, values=values, color=color, **kwargs),
    "Pie_plot": lambda df, names=None, values=None, **kwargs: px.pie(df, names=names, values=values, **kwargs),
}

class HyperParamTrial(BaseModel):
    trial_number: int
    hyperparameters: Dict[str, Any]

class MLModelConfig(BaseModel):
    model_name: str
    trials: List[HyperParamTrial]

class CleaningConfig(BaseModel):
    pipeline_type: str = Field(default="", description="Type or name of the pipeline")
    
    # === Cleaning configs ===
    remove_duplicates: bool = Field(default=True)
    scaling: bool = Field(default=True)
    normalization: bool = Field(default=True)
    handling_outliers: bool = Field(default=True)
    feature_extraction: bool = Field(default=True)
    feature_selection: bool = Field(default=True)
    
    global_nulls_handler: Optional[str] = None
    columns_nulls_handler: Optional[Dict[str, str]] = None
    columns_outliers_handler: Optional[str] = None
    removal_columns: List[str] = Field(default_factory=list)
    string_handler: Optional[str] = None

    normalization_method: Optional[str] = None
    scaling_method: Optional[str] = None
    feature_selector_method: Optional[str] = None
    feature_extraction_method: Optional[str] = None

    target_column: Optional[str] = None
    cleaned_data: Optional[Any] = None

    # === New additions (ML pipeline) ===
    task_type: Optional[str] = Field(
        default=None,
        description="ML task type (classification, regression, clustering)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Selected ML algorithm name"
    )
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hyperparameters for the selected model"
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        validate_assignment = True
                
class VisualizationConfig(BaseModel):
    """
    Configuration for visualization pipeline.
    """

    # Core visualization configs
    pipeline_type: str = Field(default="", description="Type or name of the visualization pipeline")
    chart_types: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Mapping of column names to chart types (e.g., {'sales': 'bar'})"
    )
    color_palette: Optional[str] = Field(
        default=None,
        description="Color palette to use for plots (e.g., 'viridis', 'plasma')"
    )
    show_grid: bool = Field(
        default=True,
        description="Whether to show grid lines on plots"
    )
    figure_size: Tuple[int, int] = Field(
        default=(8, 6),
        description="Figure size for plots (width, height)"
    )
    save_plots: bool = Field(
        default=False,
        description="Whether to save plots to disk"
    )
    output_format: str = Field(
        default="png",
        description="Output format for saved plots (e.g., 'png', 'jpg', 'pdf')"
    )
    generated_plots: Optional[Any] = Field(
        default=None,
        description="Holds the generated plot objects after processing"
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        validate_assignment = True  

class DataCleaningPipelineConfig:
    """
    Configuration for Data Cleaning Pipeline.
    """

    def __init__(self, config: CleaningConfig):
        self.config = config
        for k, v in config.dict().items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f"<DataCleaningPipelineConfig("
            f"pipeline_type={self.pipeline_type}, "
            f"remove_duplicates={self.remove_duplicates}, "
            f"scaling={self.scaling}, normalization={self.normalization}, "
            f"handling_outliers={self.handling_outliers}, "
            f"feature_extraction={self.feature_extraction}, "
            f"feature_selection={self.feature_selection})>"
        )
    
    def to_dict(self):
        return self.config.dict() if self.config else {}
    
class DataVisualizationPipelineConfig:
    def __init__(self, config: VisualizationConfig):
        self.config = config 
        for k, v in config.dict().items():
            if k == "output_format":   # استثناء
                self._output_format = v
            else:
                setattr(self, k, v)

    @property
    def output_format(self) -> Optional[str]:
        return self._output_format

    @output_format.setter
    def output_format(self, fmt: Optional[str]):
        allowed_formats = {"png", "jpg", "jpeg", "svg", "pdf"}
        if fmt is not None and fmt.lower() not in allowed_formats:
            raise ValueError(f"Invalid output format: {fmt}. Allowed: {allowed_formats}")
        self._output_format = fmt

    def to_dict(self):
        return self.config.dict() if self.config else {}
        
class BuildingCleaningPipeline:
    def __init__(self, config: CleaningConfig, raw_data):
        # ✅ خد نسخة dict من الـ config
        self.config_data = config.dict()
        self.data = raw_data.copy()
        self.cleaning_data = None
        self.data_matrix = None
        self.encoder = None
        self.scaler = None
        self.X = None
        self.Y = None
        self.selected_X = None
        self.selected_features = None
        self.analysis = None
        
    def data_splitting(self): 
        target_col = self.config_data.get("target_column")
        if not target_col: 
            raise ValueError(f"No target column set. Config is: {target_col}")
        
        self.Y = self.data[target_col]
        self.X = self.data.drop(columns=[target_col])  

    def handling_missing_values(self):
        nulls_config = self.config_data.get("columns_nulls_handler", None)
        target_col = self.config_data.get("target_column")

        for rule in nulls_config:
            for col, method_name in rule.items():
                is_target = (col == target_col)
                if col not in self.X.columns and not is_target:
                    continue

                handler = NULL_HANDLERS.get(method_name)
                if not handler:
                    raise ValueError(f"Unknown null handler: {method_name}")

                if method_name == "Drop Rows":
                    mask = self.Y.notna() if is_target else self.X[col].notna()
                    self.X, self.Y = self.X.loc[mask], self.Y.loc[mask]

                elif method_name == "Drop Columns":
                    if not is_target:
                        self.X = self.X.drop(columns=[col])
                    else:
                        self.Y = None

                else:
                    imputer = handler()
                    if not is_target:
                        self.X[col] = pd.Series(
                            imputer.fit_transform(self.X[[col]]).ravel(),
                            index=self.X.index
                        )
                    else:
                        self.Y = pd.Series(
                            imputer.fit_transform(self.Y.values.reshape(-1, 1)).ravel(),
                            index=self.Y.index,
                            name=self.Y.name or target_col
                        )

    def handling_missing_values_all(self):
        """
        Apply one missing value handler to all columns (X and Y).
        The config must specify a single method for all columns.
        """
        method_name = self.config_data.get("global_nulls_handler", None)
        target_col = self.config_data.get("target_column")

        if not method_name:
            return  # no global handler provided

        handler = NULL_HANDLERS.get(method_name)
        if not handler:
            raise ValueError(f"Unknown null handler: {method_name}")

        if method_name == "Drop Rows":
            combined = pd.concat([self.X, self.Y], axis=1).dropna()
            self.X, self.Y = combined.drop(columns=[target_col]), combined[target_col]

        elif method_name == "Drop Columns":
            self.X = self.X.dropna(axis=1)
            if self.Y.isna().any():
                self.Y = None

        else:
            # apply imputation separately for numeric vs categorical
            imputer = handler()

            # numeric columns
            numeric_cols = self.X.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                self.X[numeric_cols] = imputer.fit_transform(self.X[numeric_cols])

            # categorical columns
            cat_cols = self.X.select_dtypes(exclude=["number"]).columns
            if len(cat_cols) > 0:
                cat_imputer = NULL_HANDLERS.get("Most Frequent Imputation")()
                self.X[cat_cols] = cat_imputer.fit_transform(self.X[cat_cols])

            # target column if exists
            if self.Y is not None and self.Y.isna().any():
                if self.Y.dtype in ["float64", "int64"]:
                    self.Y = pd.Series(
                        imputer.fit_transform(self.Y.values.reshape(-1, 1)).ravel(),
                        index=self.Y.index,
                        name=self.Y.name or target_col
                    )
                else:
                    cat_imputer = NULL_HANDLERS.get("Most Frequent Imputation")()
                    self.Y = pd.Series(
                        cat_imputer.fit_transform(self.Y.values.reshape(-1, 1)).ravel(),
                        index=self.Y.index,
                        name=self.Y.name or target_col
                    )

    def handling_outliers(self):
        method_name = self.config_data.get("columns_outliers_handler")
        if not method_name:
            return  # مفيش حاجة نعملها

        handler = OUTLIER_HANDLERS.get(method_name)
        if not handler:
            raise ValueError(f"Unknown Outlier method: {method_name}")

        numeric_cols = self.data.select_dtypes(include=["number"]).columns

        # لو الفنكشن محتاج target_col (بنمرر عمود واحد)
        if "target_col" in handler.__code__.co_varnames:
            for col in numeric_cols:
                self.data = handler(self.data, target_col=col)
        else:
            for col in numeric_cols:
                self.data[col] = handler(self.data[[col]])

    def data_type_convertor(self):
        method_name = self.config_data.get("string_handler")
        handler_cls = LABELING_METHODS.get(method_name)
        if not handler_cls:
            raise ValueError(f"Unknown String Converter method: {method_name}")
        
        if method_name == "Label Encoder":
            for col in self.X.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                self.X[col] = pd.Series(
                    le.fit_transform(self.X[col].astype(str)),
                    index=self.X.index
                )
        else:
            # ممكن تضيف هنا OneHot أو أي ميثود تانية
            pass

        # Encode Y if needed
        if self.Y is not None and self.Y.dtype in ["object", "category"]:
            le = LabelEncoder()
            self.Y = pd.Series(le.fit_transform(self.Y.astype(str)), index=self.Y.index)

    def data_normalization(self): 
        method_name = self.config_data.get("normalization_method")
        if not method_name:
            return
        method_constructor = NORMALIZATION_METHODS[method_name]
        norm = method_constructor()
        numeric_cols = self.X.select_dtypes(include=["number"]).columns
        self.X[numeric_cols] = norm.fit_transform(self.X[numeric_cols])

    def data_scaling(self):
        method_name = self.config_data.get("scaling_method")
        if not method_name:
            return
        method_constructor = SCALERS[method_name]
        self.scaler = method_constructor()
        numeric_cols = self.X.select_dtypes(include=["number"]).columns
        self.X[numeric_cols] = self.scaler.fit_transform(self.X[numeric_cols])

    def feature_selection(self):
        method_name = self.config_data.get("feature_selector_method")  # ✅ fix key
        if not method_name:
            return
        method_constructor = FEATURE_SELECTORS[method_name]
        self.selector = method_constructor()
        numeric_cols = self.X.select_dtypes(include=["number"]).columns
        self.selected_X = self.selector.fit_transform(self.X[numeric_cols], self.Y)
        self.selected_features = numeric_cols[self.selector.get_support()]


    @performance_logger
    def clean(self):
        self.data_splitting()
        if self.config_data.get("columns_nulls_handler") is not None:
            self.handling_missing_values()
            
        if self.config_data.get("global_nulls_handler") is not None:
            self.handling_missing_values_all()
            
        if self.config_data.get("columns_outliers_handler") and self.config_data.get("handling_outliers"):
            self.handling_outliers()
        if self.config_data.get("string_handler"):
            self.data_type_convertor()
        # if self.config_data.get("remove_duplicates"):
        #     combined = pd.concat([self.X, self.Y], axis=1)
        #     combined.drop_duplicates(inplace=True)
        #     self.X, self.Y = combined.drop(columns=[self.config_data["target_column"]]), combined[self.config_data["target_column"]]
        if self.config_data.get("normalization_method"):
            self.data_normalization()
        if self.config_data.get("scaling_method"):
            self.data_scaling()
        if self.config_data.get("feature_selector_method"):
            self.feature_selection()
        self.cleaning_data = pd.concat([self.X, self.Y], axis=1)


    
    # @performance_logger
    # def clean(self):
    #     self.data_splitting()
    #     if self.config_data.get("columns_nulls_handler"):
    #         self.handling_missing_values()
    #     if self.config_data.get("global_nulls_handler"):
    #         self.handling_missing_values_all()
    #     if self.config_data.get("columns_outliers_handler") and self.config_data.get("handling_outliers"):
    #         self.handling_outliers()
    #     if self.config_data.get("string_handler"):
    #         self.data_type_convertor()
    #     if self.config_data.get("normalization_method"):
    #         self.data_normalization()
    #     if self.config_data.get("scaling_method"):
    #         self.data_scaling()
    #     if self.config_data.get("feature_selector_method"):
    #         self.feature_selection()
            
    #     # 2. البيانات المنضفة + العمود الهدف
    #     self.cleaning_data = pd.concat([self.X, self.Y], axis=1)
    #     # 3. تحديد الامتداد
    #     extension = self.config_data.get("file_type", "csv").lower()
    #     target_column = self.config_data.get("target_column")

    #     buffer = BytesIO()
    #     filename = f"cleaned_data.{extension}"

    #     # 4. حفظ البيانات في الذاكرة بالامتداد المطلوب
    #     if extension == "csv":
    #         self.cleaning_data.to_csv(buffer, index=False)
    #     elif extension == "excel":
    #         self.cleaning_data.to_excel(buffer, index=False)
    #     elif extension == "json":
    #         self.cleaning_data.to_json(buffer, orient="records")
    #     elif extension == "xml":
    #         self.cleaning_data.to_xml(buffer, index=False)
    #     else:
    #         raise ValueError(f"Unsupported file type: {extension}")

    #     buffer.seek(0)

    #     # 5. إنشاء كائن CleanedData وربطه بالتحليل
    #     cleaned_file = CleanedData.objects.create(
    #         analysis=self.analysis,
    #         target_column=target_column,
    #         size=buffer.getbuffer().nbytes,
    #         type=extension,
    #     )

    #     cleaned_file.clean_data.save(filename, ContentFile(buffer.read()))
    #     buffer.close()

    #     return cleaned_file
    
    

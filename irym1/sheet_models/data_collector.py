import pandas as pd
import json
from fastapi.encoders import jsonable_encoder
import pandas as pd
import pandas as pd
import json
from functools import wraps
import numpy as np
import math
from core_performance import performance_logger
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


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
LABELING_METHODS = {
    "Label Encoder": LabelEncoder,
    "One-Hot Encoder": lambda: OneHotEncoder(sparse=False),  # returns instance
    "Ordinal Encoder": OrdinalEncoder,
    "Count Vectorizer": CountVectorizer,          # text → counts
    "TF-IDF Vectorizer": TfidfVectorizer          # text → weighted counts
}
NORMALIZATION_METHODS = {
    "L1 Normalization": lambda: Normalizer(norm="l1"),
    "L2 Normalization": lambda: Normalizer(norm="l2"),
    "Max Normalization": lambda: Normalizer(norm="max"),
    "Min-Max Scaling (0-1)": MinMaxScaler,
    "Max-Abs Scaling": MaxAbsScaler
}


def error_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs): 
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("\n\n==========================\n")
            print(f"Function: {func.__name__}")
            print(f"Args: {args[1:] if len(args)>1 else args}")
            print(f"Kwargs: {kwargs}")
            print(f"Error: {e}")
            print("\n==========================\n\n")
            # يمكننا رفع نفس الخطأ أو تحويله لقيمة موحدة
            raise ValueError(f"Error in function '{func.__name__}': {e}")
    return wrapper


class GetData:
    __allowed_types = {
        "xml": {"xml"}, 
        "csv": {"csv"}, 
        "json": {"json"},
        "sql": {"sql"}, 
        "excel": {"xlsx", "xls", "xlsm", "xlsb", "xltx", "xltm"}
    }

    __all_extensions = set().union(*__allowed_types.values())

    def __init__(self, path: str=None,
                 type_: str=None,
                 cleaned_data_path = None,
                 target:str=None,
                 cleaned_type:str=None):
        
        self.__path = path if path else cleaned_data_path
        self.__type = type_.lower() if type_ is not None else cleaned_type.lower()
        self.__raw_data = None
        self.load_data()
        self.clean_data()  # تنظيف البيانات فور التحميل
        self.X = None
        self.y = None
        self.target = target

        
    
    
    @error_logger
    @performance_logger
    def __is_allowed(self, type_: str) -> bool:
        return type_ in self.__all_extensions

    @error_logger
    @performance_logger
    def load_data(self, sql_connection=None, sql_query=None):
        if not self.__is_allowed(self.__type):
            raise ValueError(f"File type '{self.__type}' is not supported.")

        if self.__type in self.__allowed_types["excel"]:
            self.__raw_data = pd.read_excel(self.__path)
        elif self.__type in self.__allowed_types["csv"]:
            self.__raw_data = pd.read_csv(self.__path)
        elif self.__type in self.__allowed_types["json"]:
            self.__raw_data = pd.read_json(self.__path)
        elif self.__type in self.__allowed_types["xml"]:
            self.__raw_data = pd.read_xml(self.__path)
        elif self.__type in self.__allowed_types["sql"]:
            if sql_connection is None or sql_query is None:
                raise ValueError("For SQL files, provide sql_connection and sql_query.")
            self.__raw_data = pd.read_sql(sql_query, sql_connection)
        else:
            self.__raw_data = None

        return self.__raw_data

    @error_logger
    @performance_logger
    def get_train_data(self):
        if self.__raw_data is None:
            print("⚠️ No data loaded.")
            return {"X": None, "y": None,}

        if not self.target or self.target not in self.__raw_data.columns:
            print(f"⚠️ Target column '{self.target}' not found in data.")
            return {"X": None, "y": None, }

        try:
            data = self.__raw_data.copy()
            self.y = data[self.target]
            self.X = data.drop(columns=[self.target])

            print(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")

            return {
                "X": self.X,
                "y": self.y,
            }

        except Exception as e:
            print(f"Error in get_train_data: {e}")
            return {"X": None, "y": None,}
        
    @error_logger
    @performance_logger
    def clean_data(self):
        if self.__raw_data is None:
            return
        # استبدال None و NaN و Infinity بقيمة 0
        self.__raw_data = self.__raw_data.replace([np.inf, -np.inf], 0)
        self.__raw_data = self.__raw_data.fillna(0)

    @property
    @error_logger
    @performance_logger
    def get_loaded_data(self):
        return self.__raw_data
    
    @property
    @error_logger
    @performance_logger
    def cols_names(self):
        return self.__raw_data.columns.tolist() if self.__raw_data is not None else []

    @error_logger
    @performance_logger
    def data_sample(self, amount=30, as_dict=True):
        if self.__raw_data is None:
            return None
        sample = self.__raw_data.sample(min(amount, len(self.__raw_data)))
        return sample.to_dict(orient="records") if as_dict else sample

    @property
    @error_logger
    @performance_logger
    def data_info(self):
        if self.__raw_data is None:
            return {}
        return {
            "columns": self.__raw_data.columns.tolist(),
            "dtypes": self.__raw_data.dtypes.apply(lambda x: x.name).to_dict(),
            "non_null_counts": self.__raw_data.notnull().sum().to_dict(),
            "shape": list(self.__raw_data.shape)
        }
    

    @property
    @error_logger
    @performance_logger
    def value_count(self, top_n=None):
        if self.__raw_data is None:
            return {}
        result = {}
        for col in self.__raw_data.columns:
            counts = self.__raw_data[col].value_counts()
            if top_n:
                counts = counts.head(top_n)
            result[col] = counts.to_dict()
        return result

    @property
    @error_logger
    @performance_logger
    def long_description(self):
        raw = {
            "cols": self.cols_names,
            "data_sample": self.data_sample(5),
            "data_info": self.data_info,
            "value_count": self.value_count
        }
        # JSON آمن بدون NaN أو Infinity
        return json.loads(json.dumps(raw, allow_nan=False))

    def __repr__(self):
        if self.__raw_data is None:
            return "<GetData: empty>"
        return f"<GetData shape={self.__raw_data.shape} type={self.__type}>"
    
    
    
# ---------------------------------------------
# بسم الله الرحمن الرحيم
"""
SmartSheet IRYM 1 - FastAPI Public Tunnel Setup
Developed by:
  - Mohamed Alaa
  - Abd El-Rahman Kamal
"""

# 🔹 Purpose:
# Run this script to generate a public ngrok URL for your FastAPI backend.
# Copy the generated URL and paste it into your Django .env (NG_KEY), then run the Django server.

# ================== STEP 1: Install Required Libraries (Once Only) ==================
# !pip install -qU pyngrok nest_asyncio fastapi uvicorn

# ================== STEP 2: Import Libraries ==================
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# ================== STEP 3: Apply nest_asyncio ==================
# Enables running FastAPI and ngrok in Jupyter/Colab environments
nest_asyncio.apply()

# ================== STEP 4: Configure ngrok ==================
PORT = 7860  # Change this if your FastAPI runs on a different port

# Kill any existing tunnels to avoid conflicts
ngrok.kill()

# Set your ngrok authentication token
NGROK_AUTH_TOKEN = "PUT_YOUR_NGROK_AUTH_KEY_HERE"
if NGROK_AUTH_TOKEN == "PUT_YOUR_NGROK_AUTH_KEY_HERE":
    raise ValueError("!! Please replace NGROK_AUTH_TOKEN with your actual ngrok token!")

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Open a new tunnel
public_url = ngrok.connect(PORT, bind_tls=True)
print("------------------- NG-PUBLIC KEY ---------------------")
print(f"Public FastAPI URL: {public_url}/docs")
print("Copy this URL and paste it into your Django .env as NG_KEY")
print("------------------- NG-PUBLIC KEY ---------------------")
# ----------------------------------------------------------------- 

from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from enum import Enum

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import uuid, os, torch, traceback, asyncio
import nest_asyncio
import uvicorn

import functools
import logging
import inspect

import httpx
from pydantic import BaseModel, Field, create_model, ValidationError
from typing import List, Optional,Dict,Tuple, Literal, Any

import re
import json

import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import Request
from pydantic import conlist
app = FastAPI()
logger = logging.getLogger(__name__)

# Configure logging once (you can configure handlers/formatters as needed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LLM = {"normal":"Qwen/Qwen2.5-0.5B-Instruct",
       "coder": "Qwen/Qwen2.5-Coder-3B-Instruct",
       }

model_normal = AutoModelForCausalLM.from_pretrained(
    LLM["normal"],
    torch_dtype="auto",
    device_map="auto"
)
tokenizer_normal = AutoTokenizer.from_pretrained(LLM["normal"])

coder = AutoModelForCausalLM.from_pretrained(
    LLM["coder"],
    torch_dtype="auto",
    device_map="auto"
)

tokenizer_coder = AutoTokenizer.from_pretrained(LLM["coder"])



CURRENT_WORKING = {}
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
MODEL_SYSTEM_MESSAGE_CONTENT="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

null_handler_list = Literal[
    "Drop Rows",
    "Drop Columns",
    "Mean Imputation",
    "Median Imputation",
    "Most Frequent Imputation",
    "Constant Imputation",
    "KNN Imputation",
    "Iterative Imputation"
]

normalization_method_list = Literal[
    "L1 Normalization",
    "L2 Normalization",
    "Max Normalization",
    "Min-Max Scaling (0-1)",
    "Max-Abs Scaling"
]

scaler_list = Literal[
    "Standard Scaler",
    "Min-Max Scaler",
    "Max-Abs Scaler",
    "Robust Scaler",
    "Power Transformer (Yeo-Johnson)",
    "Power Transformer (Box-Cox)",
    "Quantile Transformer (Uniform)",
    "Quantile Transformer (Normal)",
    "L2 Normalizer"
]

labeling_method_list = Literal[
    "Label Encoder",
    "One-Hot Encoder",
    "Ordinal Encoder",
    "Count Vectorizer",
    "TF-IDF Vectorizer"
]



outlier_methods = Literal["Z-Score Removal",
                          "IQR Removal",
                          "Isolation Forest",
                          "Elliptic Envelope"]

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

async def generate(messages):
    # Apply chat template
    text = tokenizer_normal.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    model_inputs = tokenizer_normal([text], return_tensors="pt").to(model_normal.device)

    # Generate output
    outputs = model_normal.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # Remove prompt part
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
    ]

    # Decode to string
    response = tokenizer_normal.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

async def generate_code(messages):
    # Apply chat template
    text = tokenizer_coder.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    model_inputs = tokenizer_coder([text], return_tensors="pt").to(coder.device)

    # Generate output
    outputs = coder.generate(
        **model_inputs,
        max_new_tokens=2048,       # أطول
        temperature=0.7,           # تنويع بسيط
        top_p=0.9,                 # nucleus sampling
        repetition_penalty=1.1     # يمنع التكرار
    )

    # Remove prompt part
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
    ]

    # Decode to string
    response = tokenizer_coder.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def create_message(*args, **kwargs):
    """Prepare the conversation for t+he LLM"""
    details = kwargs.get("details")
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Return the output as a strict JSON object that fits exactly "
                "the following Pydantic schema. Do not add extra fields or text. "
                "Only return a valid JSON."
            )
        },
        {
            "role": "user",
            "content": "\n".join([
                "",
                str(kwargs.get("message", "")),
                "",
                "## Pydantic Schema (use exactly this structure):",
                json.dumps(details.model_json_schema(), ensure_ascii=False, indent=2) if details else "",
                "",
                "### Output JSON (must match schema):"
            ])
        }
    ]

def coder_message(*args, **kwargs):
    """Prepare the conversation for LLM to generate Python code only"""
    return [
        {
            "role": "system",
            "content": (
                "You are Qwen2.5 Coder, a highly precise Python coding assistant. "
                "Your ONLY task is to generate valid and correct Python code. "
                "Follow these rules strictly: "
                "1. Return ONLY Python code, inside a valid code block (```python ... ```). "
                "2. Do not add explanations, comments, or extra text. "
                "3. Ensure the code runs without syntax errors. "
                "4. Do not return JSON, markdown outside the code block, or any prose. "
                "Your output must always be executable Python code only."
            )
        },
        {
            "role": "user",
            "content": "\n".join([
                "Write a Python script that satisfies the following requirement:",
                str(kwargs.get("message", "")),
                "",
                "IMPORTANT: Return only the Python code, inside ```python ... ```."
            ])
        }
    ]

def extract_python(raw_response: str) -> str | None:
    """
    Extracts Python code from the raw LLM response.
    - Supports fenced code blocks ```python ... ```
    - Falls back to any ```...``` block
    - If no fences: tries to detect by indentation / 'def ' / 'class ' / imports
    """
    # Remove leading/trailing spaces
    cleaned = raw_response.strip()

    # Case 1: Proper fenced block with "python"
    match = re.search(r"```python\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Case 2: Any fenced block ```
    match = re.search(r"```(.*?)```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Case 3: Try heuristic detection (fallback)
    # Look for typical Python signatures
    py_start = re.search(r"(?:^|\n)(import |from |def |class )", cleaned)
    if py_start:
        return cleaned[py_start.start():].strip()

    # Nothing found
    print("❌ No Python code detected in response.")
    return None



def extract_json(raw_response: str):
    """
    يحاول استخراج JSON صالح من رد خام للـ LLM.
    - يتجاهل أي نص بعد إغلاق JSON.
    - يصحح قيم Python-like إلى JSON صالح.
    """
    # تنظيف Markdown
    cleaned = re.sub(r"```json|```", "", raw_response).strip()

    # البحث عن أول '{'
    obj_start = cleaned.find('{')
    if obj_start == -1:
        return None

    depth = 0
    end_idx = None
    for i, ch in enumerate(cleaned[obj_start:], start=obj_start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    if end_idx is None:
        print("❌ No matching closing brace found for JSON object.")
        return None

    candidate = cleaned[obj_start:end_idx]

    # إصلاح قيم Python-like
    candidate_fixed = (
        candidate
        .replace("None", "null")
        .replace("True", "true")
        .replace("False", "false")
    )

    try:
        return json.loads(candidate_fixed)
    except json.JSONDecodeError as e:
        print("❌ JSON decode failed:", e)
        print("🔍 Candidate object:", candidate_fixed)
        return None


async def run_llm_task(response_model, specific_message):
    # 1. Build messages for LLM
    messages = create_message(
        message=specific_message,
        details=response_model
    )

    # 2. Generate raw text from LLM
    raw_response = await generate(messages)

    # 3. GPU cleanup (اختياري)
    clear_gpu()

    # 4. Extract JSON
    parsed = extract_json(raw_response)
    print(parsed)

    # 5. Validation
    if parsed:
        try:
            return response_model(**parsed)
        except Exception:
            return response_model()  # fallback للـ schema الفاضي
    else:
        raise ValueError("❌ Invalid JSON in LLM response")

async def run_coder_task_python(specific_message: str):
    # 1. Build messages
    messages = coder_message(message=specific_message)

    # 2. Generate raw text from LLM
    raw_response = await generate_code(messages)

    # 3. Cleanup GPU if needed
    clear_gpu()

    # 4. Extract Python code
    code = extract_python(raw_response)
    print("🔍 Extracted Python Code:\n", code)

    # 5. Optional: validate by trying to compile
    if code:
        try:
            compile(code, "<llm_code>", "exec")
            return code
        except SyntaxError as e:
            print("❌ Python syntax error:", e)
            return code
    return None

# Request schema
class SummaryResultRequest(BaseModel):
    prompt: str = Field(..., max_length=5000)
    cols: List[str] = Field(..., description="List of column names related to the dataset")

# Response schema
class SummaryResultResponse(BaseModel):
    data_title: str = Field(min_length=5, max_length=100, description="Data title", default="Error",)
    data_description: str = Field(
        default="Error this is a default ",
        min_length=10,
        max_length=4000,
        description="Detailed description of the dataset based on the user prompt"
    )

#Boolean Configs

# ===== ML TASK SELECTION =====
class TaskSelectionResponse(BaseModel):
    task_type: Literal["classification", "regression", "clustering"]



# ===== MODEL SELECTION =====

MODEL_NAME_MAPPING = {
    "RandomForest": ["Random Forest Classifier", "Random Forest Regressor"],
    "LogisticRegression": ["Logistic Regression"],
    "SVM": ["Support Vector Classifier", "Support Vector Regressor"],
    "KNeighbors": ["KNN Classifier", "KNN Regressor"],
    "DecisionTree": ["Decision Tree Classifier", "Decision Tree Regressor"],
    "GradientBoosting": ["Gradient Boosting Classifier", "Gradient Boosting Regressor"],
    "AdaBoost": ["AdaBoost Classifier", "AdaBoost Regressor"],
    "NaiveBayes": ["Gaussian Naive Bayes"],
    "XGBoost": ["XGBoost"],
    "LightGBM": ["LightGBM"],
    "LinearRegression": ["Linear Regression"],
    "ElasticNet": ["ElasticNet"],
    "KMeans": ["KMeans Clustering"],
    "DBSCAN": ["DBSCAN"],
}





# ===== HYPERPARAM TRIALS =====
class HyperParamTrial(BaseModel):
    """A single trial configuration of hyperparameters for a given ML model."""
    trial_number: int
    hyperparameters: Dict[str, Any]

class MLModelConfig(BaseModel):
    model_name: str
    trials: List[HyperParamTrial]



class DataCleaningBoolConfReq(BaseModel):
    columns: List[str] = Field(..., description="List of column names")
    dtypes: Dict[str, str] = Field(..., description="Mapping of column names to data types")
    non_nulls_counts: Dict[str, int] = Field(..., description="Non-null count per column")
    shape: Tuple[int, int] = Field(..., description="Shape of the dataset (rows, columns)")

class DataCleaningBoolConfRes(BaseModel):
    remove_duplicates: bool = Field(default=False, description="Remove duplicate rows")
    scaling: bool = Field(default=False, description="Apply feature scaling")
    normalization: bool = Field(default=False, description="Normalize features")
    handling_outliers: bool = Field(default=False, description="Handle outliers")
    feature_extraction: bool = Field(default=False, description="Perform feature extraction")
    feature_selection: bool = Field(default=False, description="Perform feature selection")

class Nulls_method_request(BaseModel):
  data_sample: dict = Field(..., description="data sample")

class Nulls_method_response(BaseModel):
  null_method: null_handler_list = Field(description="fit nulls method",
                                         default="Error this is a default ",)

class Outlier_method_request(BaseModel):
  data_sample: dict = Field(..., description="data sample")

class Outlier_method_response(BaseModel):
  outlier_method: outlier_methods = Field(description="fit nulls method",
                                         default="Error this is a default ",)

class Normalization_request(BaseModel):
  data_sample: dict = Field(..., description="data sample")

class Normalization_response(BaseModel):
  normalization_method: normalization_method_list = Field(description="fit nulls method", default="Error this is a default ",)

class Scaling_request(BaseModel):
  data_sample: dict = Field(..., description="data sample")

class Scaling_response(BaseModel):
  scaleing_method: scaler_list =  Field(description="fit nulls method", default="Error this is a default ",)

class Labeling_method_request(BaseModel):
  data_sample: dict = Field(..., description="data sample")

class Labeling_method_response(BaseModel):
  labeling_method: labeling_method_list = Field(description="fit nulls method", default="Error this is a default ",)

class RemoveColsReq(BaseModel):
  data_sample: dict = Field(..., description="data sample")

class ColumnDescribe(BaseModel):
    col_description: str = Field(
        default="Default description for this column",
        max_length=1500,
        min_length=10
    )

    class Config:
        extra = "forbid"  # أي field زيادة هيترف

class LongDescription(BaseModel):
    cols: list[str]
    data_sample: list[dict]
    data_info: dict
    value_count: dict

class DataCollector(BaseModel):
    prompt: str
    long_description: LongDescription

class CodeRequest(BaseModel):
    prompt: str
    sample: Optional[str] = None

class GeneratedModel(BaseModel):
    code: str

def make_enum(name: str, values: list[str]):
    return Enum(name, {v: v for v in values})

def make_target_model(cols: list[str]):
    ColsEnum = make_enum("ColsEnum", cols)
    TargetColRes = create_model(
        "TargetColRes",
        target_column=(ColsEnum, None)  # Optional[ColsEnum]
    )
    return TargetColRes

def make_cols_description(cols: list[str]):
    class ColumnDescription(BaseModel):
        col: str                 # هنسمح بأي سترينج، ونفلتر/نطابق بعدين لو حبيت
        col_description: str
    return ColumnDescription


class CurrentDatasets:
    def __init__(self):
        self.id = None
        self.info = None
        self.sample = None
        self.cols = None
        self.title = None
        self.description = None
        self.time_created = None
        self.django_request = None
        self.status = None
        self.cols_enum = None
        self.prompt = None
        self.cols_description = None

    def add_to_current(self):
        self.id = str(len(CURRENT_WORKING) + 1)
        self.time_created = datetime.datetime.now()
        CURRENT_WORKING[self.id] = self   # ← object مش dict


    def to_dict(self):
        return {
            "id": self.id,
            "info": self.info,
            "sample": self.sample,
            "cols": self.cols,
            "title": self.title,
            "description": self.description,
            "time_created": self.time_created,
            "django_request": self.django_request,
            "status": self.status
        }

    def is_finished(self):
        return self.status == "finish"

    def is_working(self):
        return self.status == "work"

    def is_waiting(self):
        return self.status == "wait"

    def delete(self):
        if self.id in CURRENT_WORKING:
            del CURRENT_WORKING[self.id]

    def set_info(self, info):
        self.info = info or self.info

    def set_sample(self, sample):
        self.sample = sample or self.sample

    def set_cols(self, cols):
        self.cols = cols or self.cols

    def set_title(self, title):
        self.title = title or self.title

    def set_prompt(self, prompt):
      self.prompt = prompt

    def set_descritpion(self, des):
        self.description = des

    def set_cols_description(self, cols_description):
        self.cols_description = cols_description

    def set_django_request(self, request):
        self.django_request = request or self.django_request

    def update_status(self, stat):
        self.status = stat or self.status

    def get_current(self, id=None):
        if id:
            if id not in CURRENT_WORKING:
                raise ValueError(f"No Data with this id: {id}")
            return CURRENT_WORKING[id]
        elif self.id:
            return CURRENT_WORKING.get(self.id)
        else:
            raise ValueError("No id provided and object has no id yet")

class CodeEngine:
    def __init__(self, prompt: str, response_model: type | None = None, sample: str | None = None, debug: bool = False):
        """
        Engine to generate Python code based on a user prompt and optional dataset/sample.

        :param prompt: The main instruction for code generation.
        :param response_model: Optional Pydantic model/schema to validate the generated code output.
        :param sample: Optional dataset or code sample to guide generation.
        :param debug: If True, prints internal messages for debugging.
        """
        self.prompt = prompt
        self.response_model = response_model
        self.sample = sample
        self.gen_code: str | None = None
        self.debug = debug

    def _build_prompt(self) -> str:
        """
        Construct a structured and optimized prompt for the code generation model.
        """
        # Base instruction
        instructions = [
            "You are a Python expert AI assistant.",
            "Generate clean, efficient, and well-commented Python code.",
            "Follow best practices and include imports if needed."
        ]

        # Add user prompt
        instructions.append(f"\nUser Instructions:\n{self.prompt}")

        # Add dataset/sample if provided
        if self.sample:
            instructions.append(f"\nDataset / Sample Reference:\n{self.sample}")

        # Add output validation instruction if response_model is provided
        if self.response_model:
            instructions.append(
                "\nEnsure the output code conforms to the following schema or model: "
                f"{self.response_model.__name__}"
            )

        # Combine all instructions
        final_prompt = "\n".join(instructions)

        if self.debug:
            print("=== Generated Prompt ===")
            print(final_prompt)
            print("=======================")

        return final_prompt

    async def _run_task(self) -> str:
        """
        Internal method to run code generation asynchronously.
        """
        messages = self._build_prompt()
        # Wrap the message in coder_message for system + user roles
        wrapped_message = coder_message(
            message=messages,
            details=self.response_model
        )

        try:
            task_result = await run_coder_task_python(specific_message=wrapped_message)
        except Exception as e:
            raise RuntimeError(f"Code generation failed: {e}") from e

        return task_result

    async def create_code(self) -> str:
        """
        Generate Python code based on the prompt and optional sample.
        Stores the generated code in self.gen_code.
        """
        self.gen_code = await self._run_task()
        return self.gen_code

# Old LLMEngine No parallelization -- it work well --
# class LLMEngine:
#     def __init__(self, current):
#         self.work_on = current
#         self._results = {}


#     async def _run_task(self, response_model: type[BaseModel], instruction: str, sample=None, max_retries=2):
#         """
#         Run a task with strict instructions for the LLM.
#         - Enforce JSON output.
#         - Include optional sample data.
#         - Validate against Pydantic model (including Literal fields).
#         """
#         specific_message = [
#             "You are an expert AI assistant. Follow the instructions strictly.",
#             "Always return VALID JSON that can be parsed programmatically.",
#             "Do NOT include explanations or markdown. Only return JSON."
#         ]
#         specific_message.append(instruction)

#         if sample is not None:
#             specific_message.append("\n## Data Sample:")
#             specific_message.append(str(sample))

#         attempt = 0
#         while attempt <= max_retries:
#             attempt += 1
#             task = await run_llm_task(
#                 response_model=response_model,
#                 specific_message="\n".join(specific_message)
#             )

#             # Validate against Pydantic model
#             try:
#                 validated = response_model.model_validate(task)  # for Pydantic v2
#                 return validated
#             except (ValidationError, AttributeError):
#                 try:
#                     validated = response_model(**task)  # fallback for v1
#                     return validated
#                 except ValidationError as e:
#                     # Retry if invalid Literal or JSON
#                     if attempt > max_retries:
#                         raise e
#                     # Optionally modify prompt to enforce valid Literal
#                     specific_message.append(
#                         "The previous output was invalid. Please respond using ONLY the exact allowed values from the model's Literal."
#                     )
#                     await asyncio.sleep(0.5)  # short delay before retry


#     async def data_info_summary(self):
#         data = self.work_on
#         specific_message = [
#             "## User Prompt:",
#             str(getattr(data, "prompt", "")),
#             "",
#             "## Data columns:",
#             "Give a description for this dataset based on the user prompt and the columns:",
#             str(data.cols),
#             "This is user prompt use it if needed",
#             str(data.prompt),
#         ]
#         res =  await self._run_task(
#             response_model=SummaryResultResponse,
#             instruction="\n".join(specific_message)
#         )
#         data.set_descritpion(res)
#         return res

#     async def select_target(self):
#         data = self.work_on
#         TargetColRes = make_target_model(data.cols)

#         # Build a clear and strict prompt for the LLM
#         specific_message = [
#             "## Task:",
#             "You are an expert in data analysis and machine learning.",
#             "Select the most appropriate target column for prediction in the dataset.",
#             "",
#             "Rules:",
#             "- If the user specified a target column and it exists in the dataset, use it.",
#             "- Otherwise, select the column that is most suitable for prediction.",
#             "- Return ONLY a JSON object with one key: `target_column`.",
#             "- Do NOT include explanations, comments, or markdown.",
#             "",
#             "Dataset columns:",
#             str(data.cols),
#         ]

#         # Include user prompt if available
#         if getattr(data, "prompt", None):
#             specific_message.append(f"User instruction: {data.prompt}")

#         # Join the instructions
#         instruction_text = "\n".join(specific_message)

#         # Run the LLM task
#         return await self._run_task(
#             response_model=TargetColRes,
#             instruction=instruction_text
#         )

#     async def columns_description(self):
#         data = self.work_on
#         cols = data.cols or []
#         cols_descriptions = []  # list of dicts instead of {"descriptions": []}

#         data_des = data.description or ""
#         print(f"========================================\nDataset description: {data_des}\n\n")

#         for col in cols:
#             specific_message = [
#                 "## Task:",
#                 "You are an expert data analyst.",
#                 "Provide a concise and informative description for a single dataset column.",
#                 "",
#                 "Output requirements:",
#                 "- Return ONLY a JSON object with EXACTLY one key: `col_description`.",
#                 "- Do NOT add explanations, comments, or markdown.",
#                 "- Do NOT include any other fields.",
#                 "- Do NOT duplicate the key `col_description`.",
#                 "",
#                 "Rules for description:",
#                 "- Focus ONLY on the given column.",
#                 "- Maximum 3 sentences.",
#                 "- Each column description must be UNIQUE.",
#                 "- Avoid repeating phrases across multiple columns.",
#                 "",
#                 "Example of valid output:",
#                 '{ "col_description": "This column stores categorical data about product categories." }',
#                 "",
#                 f"Column name (use EXACT name): {col}",
#                 f"Dataset description: {data_des}",
#             ]

#             # Include user prompt if available
#             if getattr(data, "prompt", None):
#                 specific_message.insert(1, f"## User instruction: {data.prompt}")

#             instruction_text = "\n".join(specific_message)

#             # Run the LLM task
#             res = await self._run_task(
#                 response_model=ColumnDescribe,
#                 instruction=instruction_text
#             )

#             # Parse result (Pydantic V2 fallback to V1)
#             try:
#                 parsed = res.model_dump()
#             except AttributeError:
#                 parsed = res.dict()

#             # Attach column description
#             cols_descriptions.append({
#                 "column": col,
#                 "description": parsed.get("col_description", "")
#             })

#         data.set_cols_description(cols_descriptions)
#         return cols_descriptions


#     async def select_droped_cols(self):
#         pass


#     async def normalization(self):
#         instruction = "\n".join([
#             "## Task:",
#             "You are a data preprocessing expert.",
#             "Select the most appropriate normalization method for the given dataset sample.",
#             "",
#             "Rules:",
#             "- Return ONLY JSON: { 'normalization_method': '<method_name>' }",
#             "- Do NOT include explanations, comments, or markdown.",
#             "- Choose a method suitable for the given data sample.",
#             "",
#             f"User instruction (if any): {self.work_on.prompt}"
#         ])
#         return await self._run_task(
#             Normalization_response,
#             instruction,
#             self.work_on.sample
#         )

#     async def scaling(self):
#         instruction = "\n".join([
#             "## Task:",
#             "You are a data preprocessing expert.",
#             "Select the most appropriate scaler method for the given dataset sample.",
#             "",
#             "Rules:",
#             "- Return ONLY JSON: { 'scaling_method': '<method_name>' }",
#             "- Do NOT include explanations, comments, or markdown.",
#             "- Choose a method suitable for the given data sample.",
#             "",
#             f"User instruction (if any): {self.work_on.prompt}"
#         ])
#         return await self._run_task(
#             Scaling_response,
#             instruction,
#             self.work_on.sample
#         )

#     async def labeling(self):
#         instruction = "\n".join([
#             "## Task:",
#             "You are a data preprocessing expert.",
#             "Select the most appropriate labeling method for the given dataset sample.",
#             "",
#             "Rules:",
#             "- Return ONLY JSON: { 'labeling_method': '<method_name>' }",
#             "- Do NOT include explanations, comments, or markdown.",
#             "- Choose a method suitable for the given data sample.",
#             "",
#             f"User instruction (if any): {self.work_on.prompt}"
#         ])
#         return await self._run_task(
#             Labeling_method_response,
#             instruction,
#             self.work_on.sample
#         )

#     async def nulls_method(self):
#         instruction = "\n".join([
#             "## Task:",
#             "You are a data preprocessing expert.",
#             "Select the most appropriate method to handle null/missing values in the given dataset sample.",
#             "",
#             "Rules:",
#             "- Return ONLY JSON: { 'nulls_method': '<method_name>' }",
#             "- Do NOT include explanations, comments, or markdown.",
#             "- Choose a method suitable for the given data sample.",
#             "",
#             f"User instruction (if any): {self.work_on.prompt}"
#         ])
#         return await self._run_task(
#             Nulls_method_response,
#             instruction,
#             self.work_on.sample
#         )

#     async def outliers_method(self):
#         allowed_values = ["IQR", "Z-Score", "None", "Percentile"]  # ضع القيم الموجودة في Literal هنا
#         instruction = "\n".join([
#             "## Task:",
#             "You are a data preprocessing expert.",
#             "Select the most appropriate method to remove or handle outliers in the given dataset sample.",
#             "",
#             "Rules:",
#             f"- Return ONLY JSON: {{ 'outliers_method': '<method_name>' }}",
#             "- Do NOT include explanations, comments, or markdown.",
#             "- Choose ONE method suitable for the given data sample.",
#             f"- The <method_name> MUST be one of these EXACT values: {allowed_values}",
#             "",
#             f"User instruction (if any): {self.work_on.prompt}"
#         ])

#         return await self._run_task(
#             Outlier_method_response,
#             instruction,
#             self.work_on.sample
#         )



#     def get_results(self):
#         return self._results

#     async def cleaning_bool_configs(self):
#         data = self.work_on.info
#         specific_message = [
#             "## Task:",
#             "Provide boolean configs (True/False) for the following dataset info.",
#             "",
#             "## Columns:", str(data["columns"]),
#             "## Dtypes:", str(data["dtypes"]),
#             "## Non-null counts:", str(data["non_null_counts"]),
#             "## Shape:", str(data["shape"]),
#             "This is user prompt use it if needed",
#             str(self.work_on.prompt),
#         ]
#         return await run_llm_task(
#             response_model=DataCleaningBoolConfRes,
#             specific_message="\n".join(specific_message)
#         )

#     async def start(self):
#         tasks = await asyncio.gather(
#             self.data_info_summary(),
#             self.columns_description(),
#             self.normalization(),
#             self.scaling(),
#             self.nulls_method(),
#             self.outliers_method(),
#             self.cleaning_bool_configs(),
#             self.labeling(),
#             self.select_target(),
#             return_exceptions=True
#         )

#         (summary, cols_desc, norm, scale, nulls, outliers, bools, labeling, target) = tasks

#         # فكّ الليست من الـ wrapper لو كلّه تمام
#         try:
#             if isinstance(cols_desc, BaseModel) and hasattr(cols_desc, "items"):
#                 cols_desc = cols_desc.items
#         except Exception:
#             pass

#         keys = [
#             "summary_result",
#             "columns_description",
#             "normalization_method",
#             "scaling_method",
#             "nulls_method",
#             "outliers_method",
#             "bool_configs",
#             "labeling_method",
#             "target_column",
#         ]

#         self._results = dict(zip(keys, [summary, cols_desc, norm, scale, nulls, outliers, bools, labeling, target]))
#         print(self._results)


#     def results(self):
#         return self._results

# New LLMEngine with asyncio.gather for
# parallel




ML_MODELS = {
    "classification": {
        "Logistic Regression": {"max_iter": 1000, "random_state": 42},
        "Random Forest Classifier": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "Gradient Boosting Classifier": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
        "Support Vector Classifier": {"C": 1.0, "kernel": "rbf", "random_state": 42},
        "KNN Classifier": {},
        "Decision Tree Classifier": {},
        "AdaBoost Classifier": {},
        "Gaussian Naive Bayes": {},
        "XGBoost": {"random_state": 42} ,
        "LightGBM": {"random_state": 42} ,
    },
    "regression": {
        "Linear Regression": {},
        "Random Forest Regressor": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "Gradient Boosting Regressor": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
        "Ridge": {"alpha": 1.0, "random_state": 42},
        "Support Vector Regressor": {},
        "KNN Regressor": {},
        "Decision Tree Regressor": {},
        "AdaBoost Regressor": {},
        "ElasticNet": {},
        "XGBoost": {"random_state": 42} ,
        "LightGBM": {"random_state": 42} ,
    },
    "clustering": {
        "KMeans Clustering": {"n_clusters": 3, "random_state": 42, "n_init": 10},
        "DBSCAN": {},
    }
}

# ========== SCHEMAS ==========

TaskTypeLiteral = Literal["classification", "regression", "clustering"]
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, validator


# ====== Model Names ======
ModelNameLiteral = Literal[
    # Classification
    "Logistic Regression",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "Support Vector Classifier",
    "KNN Classifier",
    "Decision Tree Classifier",
    "AdaBoost Classifier",
    "Gaussian Naive Bayes",
    "XGBoost",
    "LightGBM",

    # Regression
    "Linear Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Ridge",
    "Support Vector Regressor",
    "KNN Regressor",
    "Decision Tree Regressor",
    "AdaBoost Regressor",
    "ElasticNet",

    # Clustering
    "KMeans Clustering",
    "DBSCAN",
]


# ====== Normalization ======
MODEL_ALIASES = {
    # Classification
    "RandomForest": "Random Forest Classifier",
    "LogisticRegression": "Logistic Regression",
    "SVM": "Support Vector Classifier",
    "KNeighbors": "KNN Classifier",
    "DecisionTree": "Decision Tree Classifier",
    "GradientBoosting": "Gradient Boosting Classifier",
    "AdaBoost": "AdaBoost Classifier",
    "NaiveBayes": "Gaussian Naive Bayes",

    # Regression
    "LinearRegression": "Linear Regression",
    "RandomForestRegressor": "Random Forest Regressor",
    "GradientBoostingRegressor": "Gradient Boosting Regressor",
    "SVR": "Support Vector Regressor",
    "KNNRegressor": "KNN Regressor",
    "DecisionTreeRegressor": "Decision Tree Regressor",
    "AdaBoostRegressor": "AdaBoost Regressor",
    "ElasticNetRegressor": "ElasticNet",

    # Clustering
    "KMeans": "KMeans Clustering",
    "DBSCAN": "DBSCAN",
}


def normalize_model_name(name: str) -> Optional[ModelNameLiteral]:
    """Normalize alias/short form to official ModelNameLiteral."""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]  # alias → official
    if name in ModelNameLiteral.__args__:
        return name  # already official
    return None  # invalid


# ====== Schemas ======
class ModelSelectionResponse(BaseModel):
    model_name: str  # raw name from LLM

    @validator("model_name")
    def normalize(cls, v):
        fixed = normalize_model_name(v)
        if not fixed:
            raise ValueError(f"Invalid model name: {v}")
        return fixed


class HyperParamTrial(BaseModel):
    task_type: Literal["classification", "regression", "clustering"]
    model_name: ModelNameLiteral
    hyperparameters: Dict[str, Any]

class MLExperimentSchema(BaseModel):
    task_type: TaskTypeLiteral
    model_name: ModelNameLiteral
    hyperparameters: Dict[str, Any]


class LLMEngine:
    def __init__(self, current):
        self.work_on = current
        self._results = {}

    async def _run_task(self, response_model: type[BaseModel], instruction: str, sample=None, max_retries=2):
        specific_message = [
            "You are an expert AI assistant. Follow the instructions strictly.",
            "Always return VALID JSON that can be parsed programmatically.",
            "Do NOT include explanations or markdown. Only return JSON."
        ]
        specific_message.append(instruction)

        if sample is not None:
            specific_message.append("\n## Data Sample:")
            specific_message.append(str(sample))

        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            task = await run_llm_task(
                response_model=response_model,
                specific_message="\n".join(specific_message)
            )
            try:
                validated = response_model.model_validate(task)
                return validated
            except (ValidationError, AttributeError):
                try:
                    validated = response_model(**task)
                    return validated
                except ValidationError as e:
                    if attempt > max_retries:
                        raise e
                    specific_message.append(
                        "The previous output was invalid. Please respond using ONLY the exact allowed values from the model's Literal."
                    )
                    await asyncio.sleep(0.5)

    async def data_info_summary(self):
        data = self.work_on
        specific_message = [
            "## User Prompt:",
            str(getattr(data, "prompt", "")),
            "",
            "## Data columns:",
            "Give a description for this dataset based on the user prompt and the columns:",
            str(data.cols),
            "This is user prompt use it if needed",
            str(data.prompt),
        ]
        res = await self._run_task(
            response_model=SummaryResultResponse,
            instruction="\n".join(specific_message)
        )
        data.set_descritpion(res)
        return res

    async def select_target(self):
        data = self.work_on
        TargetColRes = make_target_model(data.cols)
        specific_message = [
            "## Task:",
            "You are an expert in data analysis and machine learning.",
            "Select the most appropriate target column for prediction in the dataset.",
            "",
            "Rules:",
            "- If the user specified a target column and it exists in the dataset, use it.",
            "- Otherwise, select the column that is most suitable for prediction.",
            "- Return ONLY a JSON object with one key: `target_column`.",
            "- Do NOT include explanations, comments, or markdown.",
            "",
            "Dataset columns:",
            str(data.cols),
        ]
        if getattr(data, "prompt", None):
            specific_message.append(f"User instruction: {data.prompt}")

        return await self._run_task(
            response_model=TargetColRes,
            instruction="\n".join(specific_message)
        )

    async def _describe_single_column(self, col, data_des, user_prompt):
        specific_message = [
            "## Task:",
            "You are an expert data analyst.",
            "Provide a concise and informative description for a single dataset column.",
            "",
            "Output requirements:",
            "- Return ONLY a JSON object with EXACTLY one key: `col_description`.",
            "- Do NOT add explanations, comments, or markdown.",
            "- Do NOT include any other fields.",
            "- Do NOT duplicate the key `col_description`.",
            "",
            "Rules for description:",
            "- Focus ONLY on the given column.",
            "- Maximum 3 sentences.",
            "- Each column description must be UNIQUE.",
            "- Avoid repeating phrases across multiple columns.",
            "",
            f"Column name (use EXACT name): {col}",
            f"Dataset description: {data_des}",
        ]

        if user_prompt:
            specific_message.insert(1, f"## User instruction: {user_prompt}")

        instruction_text = "\n".join(specific_message)

        res = await self._run_task(
            response_model=ColumnDescribe,
            instruction=instruction_text
        )
        try:
            parsed = res.model_dump()
        except AttributeError:
            parsed = res.dict()
        return {"column": col, "description": parsed.get("col_description", "")}

    async def columns_description(self):
        data = self.work_on
        cols = data.cols or []
        data_des = data.description or ""
        print(f"========================================\nDataset description: {data_des}\n\n")
        tasks = [
            self._describe_single_column(col, data_des, getattr(data, "prompt", None))
            for col in cols
        ]
        cols_descriptions = await asyncio.gather(*tasks, return_exceptions=False)
        data.set_cols_description(cols_descriptions)
        return cols_descriptions

    async def select_droped_cols(self):
        pass

    async def normalization(self):
        instruction = "\n".join([
            "## Task:",
            "You are a data preprocessing expert.",
            "Select the most appropriate normalization method for the given dataset sample.",
            "",
            "Rules:",
            "- Return ONLY JSON: { 'normalization_method': '<method_name>' }",
            "- Do NOT include explanations, comments, or markdown.",
            "- Choose a method suitable for the given data sample.",
            "",
            f"User instruction (if any): {self.work_on.prompt}"
        ])
        return await self._run_task(
            Normalization_response,
            instruction,
            self.work_on.sample
        )

    async def scaling(self):
        instruction = "\n".join([
            "## Task:",
            "You are a data preprocessing expert.",
            "Select the most appropriate scaler method for the given dataset sample.",
            "",
            "Rules:",
            "- Return ONLY JSON: { 'scaling_method': '<method_name>' }",
            "- Do NOT include explanations, comments, or markdown.",
            "- Choose a method suitable for the given data sample.",
            "",
            f"User instruction (if any): {self.work_on.prompt}"
        ])
        return await self._run_task(
            Scaling_response,
            instruction,
            self.work_on.sample
        )

    async def labeling(self):
        instruction = "\n".join([
            "## Task:",
            "You are a data preprocessing expert.",
            "Select the most appropriate labeling method for the given dataset sample.",
            "",
            "Rules:",
            "- Return ONLY JSON: { 'labeling_method': '<method_name>' }",
            "- Do NOT include explanations, comments, or markdown.",
            "- Choose a method suitable for the given data sample.",
            "",
            f"User instruction (if any): {self.work_on.prompt}"
        ])
        return await self._run_task(
            Labeling_method_response,
            instruction,
            self.work_on.sample
        )

    async def nulls_method(self):
        instruction = "\n".join([
            "## Task:",
            "You are a data preprocessing expert.",
            "Select the most appropriate method to handle null/missing values in the given dataset sample.",
            "",
            "Rules:",
            "- Return ONLY JSON: { 'nulls_method': '<method_name>' }",
            "- Do NOT include explanations, comments, or markdown.",
            "- Choose a method suitable for the given data sample.",
            "",
            f"User instruction (if any): {self.work_on.prompt}"
        ])
        return await self._run_task(
            Nulls_method_response,
            instruction,
            self.work_on.sample
        )

    async def outliers_method(self):
        allowed_values = ["IQR", "Z-Score", "None", "Percentile"]
        instruction = "\n".join([
            "## Task:",
            "You are a data preprocessing expert.",
            "Select the most appropriate method to remove or handle outliers in the given dataset sample.",
            "",
            "Rules:",
            f"- Return ONLY JSON: {{ 'outliers_method': '<method_name>' }}",
            "- Do NOT include explanations, comments, or markdown.",
            "- Choose ONE method suitable for the given data sample.",
            f"- The <method_name> MUST be one of these EXACT values: {allowed_values}",
            "",
            f"User instruction (if any): {self.work_on.prompt}"
        ])
        return await self._run_task(
            Outlier_method_response,
            instruction,
            self.work_on.sample
        )

    def get_results(self):
        return self._results

    async def cleaning_bool_configs(self):
        data = self.work_on.info
        specific_message = [
            "## Task:",
            "Provide boolean configs (True/False) for the following dataset info.",
            "",
            "## Columns:", str(data["columns"]),
            "## Dtypes:", str(data["dtypes"]),
            "## Non-null counts:", str(data["non_null_counts"]),
            "## Shape:", str(data["shape"]),
            "This is user prompt use it if needed",
            str(self.work_on.prompt),
        ]
        return await run_llm_task(
            response_model=DataCleaningBoolConfRes,
            specific_message="\n".join(specific_message)
        )



    # ===== Inside the class =====
    async def select_ml_algorithm(self, task_type: str):
        """Select the most suitable algorithm for the chosen task type."""
        allowed_models = list(ML_MODELS[task_type].keys())

        instruction = "\n".join([
            "## Task:",
            "You are an expert ML engineer.",
            f"Select the most appropriate ML model for {task_type}.",
            "",
            f"- Allowed models: {allowed_models}",
            "- Return ONLY JSON: { 'model_name': '<model>' }",
            "- Do NOT include explanations or markdown.",
            "",
            f"Dataset columns: {self.work_on.cols}",
            f"User instruction (if any): {self.work_on.prompt}"
        ])

        return await self._run_task(ModelSelectionResponse, instruction, self.work_on.sample)


    async def generate_hyperparams(self, task_type: str, model_name: str):
        """Generate a single hyperparameter configuration for the chosen model."""
        allowed_models = list(ML_MODELS[task_type].keys())

        instruction = "\n".join([
            "## Task:",
            f"You are an expert ML engineer. Generate one hyperparameter trial for the model '{model_name}' in task '{task_type}'.",
            "",
            "Rules:",
            f"- task_type MUST be one of: ['classification', 'regression', 'clustering']",
            f"- model_name MUST be EXACTLY one of: {allowed_models}",
            "- Output MUST strictly follow this schema:",
            "{ 'task_type': '<classification|regression|clustering>', 'model_name': '<model>', 'hyperparameters': { ... } }",
            "- Only ONE trial, no lists.",
            "- hyperparameters must be valid JSON (key-value pairs).",
            "- Do NOT include explanations, comments, or markdown."
        ])

        raw = await self._run_task(MLExperimentSchema, instruction, self.work_on.sample)

        # ✅ Fallback normalization
        fixed_name = normalize_model_name(getattr(raw, "model_name", None))
        if fixed_name:
            raw.model_name = fixed_name

        return raw


    async def select_ml_task(self):
        """Ask the LLM to decide the most suitable ML task type."""
        allowed_tasks = ["classification", "regression", "clustering"]

        instruction = "\n".join([
            "## Task:",
            "You are an expert ML engineer.",
            "Analyze the dataset and decide the most appropriate ML task type.",
            "",
            f"- Allowed task types (must pick exactly one): {allowed_tasks}",
            "- Return ONLY JSON: { 'task_type': '<classification|regression|clustering>' }",
            "- Do NOT include explanations or markdown.",
            "",
            f"Dataset columns: {self.work_on.cols}",
            f"User instruction (if any): {self.work_on.prompt}"
        ])
        return await self._run_task(TaskSelectionResponse, instruction, self.work_on.sample)


    async def start(self):
        """
        Execute the full LLM pipeline for dataset analysis and ML preparation.
        """
        tasks_coros = [
            self.data_info_summary(),
            self.columns_description(),
            self.normalization(),
            self.scaling(),
            self.nulls_method(),
            self.outliers_method(),
            self.cleaning_bool_configs(),
            self.labeling(),
            self.select_target(),
            self.select_ml_task(),
        ]

        try:
            results_list = await asyncio.gather(*tasks_coros)
        except Exception as e:
            raise RuntimeError(f"Error during basic task execution: {e}")

        (
            summary,
            cols_desc,
            norm,
            scale,
            nulls,
            outliers,
            bools,
            labeling,
            target,
            task_type
        ) = results_list

        # ======= 2. Type checks =======
        if not isinstance(task_type, TaskSelectionResponse):
            raise TypeError(f"Expected TaskSelectionResponse, got {type(task_type)}")
        if not isinstance(target, BaseModel):
            raise TypeError(f"Expected BaseModel for target_column, got {type(target)}")

        # ======= 3. Select ML model =======
        model_res = await self.select_ml_algorithm(task_type.task_type)
        if not isinstance(model_res, ModelSelectionResponse):
            raise TypeError(f"Expected ModelSelectionResponse, got {type(model_res)}")

        # ======= 4. Generate hyperparameters (single trial) =======
        hyperparams_schema: MLExperimentSchema = await self.generate_hyperparams(
            task_type.task_type,
            model_res.model_name
        )
        if not isinstance(hyperparams_schema, MLExperimentSchema):
            raise TypeError(f"Expected MLExperimentSchema, got {type(hyperparams_schema)}")

        # ======= 5. Compile results =======
        self._results = {
            "summary_result": summary,
            "columns_description": cols_desc,
            "normalization_method": norm,
            "scaling_method": scale,
            "nulls_method": nulls,
            "outliers_method": outliers,
            "bool_configs": bools,
            "labeling_method": labeling,
            "target_column": target,
            "task_type": task_type,
            "model": model_res,
            "hyperparams": [
                {
                    "trial_number": 1,
                    "hyperparameters": hyperparams_schema.hyperparameters
                }
            ]
        }

        print("=== LLMEngine Start Complete ===")
        print(f"Task type: {task_type.task_type}")
        print(f"Selected model: {model_res.model_name}")
        print(f"Hyperparameters: {hyperparams_schema.hyperparameters}")

    @property
    def results(self):
        return self._results

class MLConfigEngine:
    def __init__(self, current_task_type: str, available_models: dict):
        """
        Args:
            current_task_type: One of "classification", "regression", "clustering"
            available_models: dictionary of supported ML models and default hyperparams
        """
        self.task_type = current_task_type
        self.available_models = available_models
        self._results = {}

    async def _run_task(self, response_model: type[BaseModel], instruction: str, sample=None, max_retries=2):
        """Reusable method to talk with the LLM and validate output against schema."""
        specific_message = [
            "You are an expert ML engineer. Follow the instructions strictly.",
            "Always return VALID JSON that matches the schema.",
            "Do NOT include explanations or markdown. Only return JSON.",
            instruction
        ]

        if sample is not None:
            specific_message.append("\n## Data Sample:")
            specific_message.append(str(sample))

        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            task = await run_llm_task(
                response_model=response_model,
                specific_message="\n".join(specific_message)
            )
            try:
                validated = response_model.model_validate(task)
                return validated
            except (ValidationError, AttributeError):
                try:
                    validated = response_model(**task)
                    return validated
                except ValidationError as e:
                    if attempt > max_retries:
                        raise e
                    specific_message.append(
                        "The previous output was invalid. Please ONLY use allowed values from the schema."
                    )
                    await asyncio.sleep(0.5)

    async def generate_ml_configs(self, sample=None):
        """
        Generate ML model configurations (with hyperparameter trials) for the given task_type.
        """
        allowed_models = [list(m.keys())[0] for m in self.available_models.get(self.task_type, [])]

        instruction = "\n".join([
            "## Task:",
            f"You are an expert ML engineer. Generate ML configs for {self.task_type} task.",
            "",
            "Rules:",
            "- Use ONLY the allowed models for this task.",
            f"- Allowed models: {allowed_models}",
            "- Each model must include at least one trial with hyperparameters.",
            "- Output MUST strictly follow the Pydantic schema.",
            "- Do NOT include explanations or markdown.",
            "",
            "Output should include:",
            "- task_type",
            "- models -> model_name, trials -> trial_number, hyperparameters",
        ])

        res = await self._run_task(
            response_model=MLExperimentSchema,
            instruction=instruction,
            sample=sample
        )

        self._results = res
        return res

    @property
    def results(self):
        return self._results


def api_logger(level=logging.INFO, log_return=False):
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.log(level, "="*30)
            logger.log(level, f"Function: {func.__name__}")
            logger.log(level, f"Args: {args}")
            logger.log(level, f"Kwargs: {kwargs}")
            logger.log(level, "="*30)

            result = await func(*args, **kwargs)

            if log_return:
                logger.log(level, f"Return: {result}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.log(level, "="*30)
            logger.log(level, f"Function: {func.__name__}")
            logger.log(level, f"Args: {args}")
            logger.log(level, f"Kwargs: {kwargs}")
            logger.log(level, "="*30)

            result = func(*args, **kwargs)

            if log_return:
                logger.log(level, f"Return: {result}")

            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# =======================================================================================================
# Data Collector and Generate Configs for DataCleaningPipeLine

@api_logger()
@app.post("/get_data_info/")
async def get_data_info(data: DataCollector):
    try:
        new_data = CurrentDatasets()
        new_data.set_prompt(data.prompt)
        new_data.set_cols(data.long_description.cols)
        new_data.set_sample(data.long_description.data_sample)
        new_data.set_info(data.long_description.data_info)
        new_data.add_to_current()
        new_data.cols_enum = make_enum(f"enmu_for_data{new_data.id}", data.long_description.cols)
        return {"message": "get data info done", "status": 200, "data_id": new_data.id}
    except Exception as e:
        return {"message": f"something went wrong :: {e}", "status": 500}

@api_logger()
@app.post("/get_data_config/")
async def get_data_config(request: Request):
    try:
        data = await request.json()
        id = data.get("id")

        if id:
            current = CURRENT_WORKING[str(id)]
            engine = LLMEngine(current)
            await engine.start()
            current.delete()
            results = engine.get_results()

            return {
                "message": f"get data config done successfully for data with id {id}",
                "status": 200,
                "results": results
            }
        else:
            print("id is wrong")
            return {"message": "id is wrong or empty", "status": 504}
    except Exception as e:
        print(e)
        return {"message": f"something went wrong :: {e}", "status": 500}



@app.post("/generate_code/")
async def generate_code_endpoint(req: CodeRequest):
    try:
        engine = CodeEngine(req.prompt)
        result = await engine.create_code()
        return {"code": result, "status":"200"}
    except Exception as e:
        print(e)
        return {"status":500, "message": f"Error: {e}"}




class MLModelConfig(BaseModel):
    """Configuration for one ML model including trials."""
    model_name: str
    trials: List[HyperParamTrial]



    
    
# ========== ENDPOINTS ==========
# @app.post("/generate-ml-configs/")
# async def generate_configs(req: ConfigRequest):
#     """توليد ML Configs باستخدام LLM بناءً على الموديل المطلوب"""
#     task_type = req.task_type.lower()

#     if task_type not in ML_MODELS:
#         raise HTTPException(status_code=400, detail="Invalid task_type")

#     # التأكد أن الموديل موجود في المجموعة
#     allowed_models = [list(m.keys())[0] for m in ML_MODELS[task_type]]
#     if req.model_name not in allowed_models:
#         raise HTTPException(status_code=400, detail=f"Model {req.model_name} not allowed for {task_type}")

#     # تشغيل المحرك
#     engine = MLConfigEngine(task_type, ML_MODELS)
#     result = await engine.generate_ml_configs(sample=req.sample)

#     return result.model_dump()


# =======================================================================================================
# Data Collector and Generate Configs for ModelTrainingPipeline
# Abd Al Rahman Code

# class DatasetInfo(BaseModel):
#     """
#     Dataset information model
#     """
#     num_rows: int = Field(..., description="Number of rows in dataset")
#     num_columns: int = Field(..., description="Number of columns in dataset")
#     column_names: List[str] = Field(..., description="List of column names")
#     column_types: Dict[str, str] = Field(..., description="Column data types")
#     target_column: Optional[str] = Field(None, description="Target column name")
#     has_missing_values: bool = Field(False, description="Whether dataset has missing values")
#     pipeline_type: str = Field(..., description="Type of ML pipeline")

# # class ConfigRequest2(BaseModel):
#     """
#     Configuration request model
#     """
#     dataset_info: DatasetInfo
#     user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
#     project_name: str = Field(..., description="Name of the project")
#     project_description: Optional[str] = Field(None, description="Project description")

# class MLConfig(BaseModel):
#     """
#     Generated ML configuration model
#     """
#     algorithm: str = Field(..., description="Recommended algorithm")
#     hyperparameters: Dict[str, Any] = Field(..., description="Recommended hyperparameters")
#     reasoning: str = Field(..., description="Reasoning for algorithm choice")
#     validation_split: float = Field(0.2, description="Validation split ratio")
#     cross_validation_folds: int = Field(5, description="Number of CV folds")
#     use_cross_validation: bool = Field(True, description="Whether to use CV")
#     random_seed: int = Field(42, description="Random seed for reproducibility")

# class EvaluationConfig(BaseModel):
#     """
#     Generated evaluation configuration model
#     """
#     metrics: List[str] = Field(..., description="Recommended evaluation metrics")
#     generate_plots: bool = Field(True, description="Whether to generate plots")
#     include_feature_importance: bool = Field(True, description="Include feature importance")
#     export_formats: List[str] = Field(["json", "html"], description="Export formats")
#     use_cross_validation: bool = Field(True, description="Use CV for evaluation")
#     cv_folds: int = Field(5, description="Number of CV folds")

# class ConfigResponse(BaseModel):
#     """
#     Complete configuration response model
#     """
#     ml_config: MLConfig
#     evaluation_config: EvaluationConfig
#     preprocessing_suggestions: List[str] = Field(default_factory=list)
#     warnings: List[str] = Field(default_factory=list)
#     timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# class ConfigGenerator:
#     """
#     ML Configuration Generator using rule-based recommendations
#     """

#     # Algorithm recommendations based on dataset characteristics
#     CLASSIFICATION_ALGORITHMS = {
#         "small_dataset": {
#             "algorithm": "Logistic Regression",
#             "hyperparameters": {"max_iter": 1000, "random_state": 42},
#             "reasoning": "Logistic Regression works well with small datasets and provides interpretable results"
#         },
#         "medium_dataset": {
#             "algorithm": "Random Forest Classifier",
#             "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
#             "reasoning": "Random Forest handles medium-sized datasets well and provides feature importance"
#         },
#         "large_dataset": {
#             "algorithm": "Gradient Boosting Classifier",
#             "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
#             "reasoning": "Gradient Boosting often performs well on large datasets with complex patterns"
#         },
#         "high_dimensional": {
#             "algorithm": "Support Vector Classifier",
#             "hyperparameters": {"C": 1.0, "kernel": "rbf", "random_state": 42},
#             "reasoning": "SVM works well with high-dimensional data"
#         }
#     }

#     REGRESSION_ALGORITHMS = {
#         "small_dataset": {
#             "algorithm": "Linear Regression",
#             "hyperparameters": {},
#             "reasoning": "Linear Regression is suitable for small datasets and provides interpretability"
#         },
#         "medium_dataset": {
#             "algorithm": "Random Forest Regressor",
#             "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
#             "reasoning": "Random Forest Regressor handles medium-sized datasets well with good performance"
#         },
#         "large_dataset": {
#             "algorithm": "Gradient Boosting Regressor",
#             "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
#             "reasoning": "Gradient Boosting often achieves excellent performance on large regression datasets"
#         },
#         "regularization_needed": {
#             "algorithm": "Ridge Regression",
#             "hyperparameters": {"alpha": 1.0, "random_state": 42},
#             "reasoning": "Ridge Regression helps prevent overfitting when regularization is needed"
#         }
#     }

#     CLUSTERING_ALGORITHMS = {
#         "default": {
#             "algorithm": "KMeans Clustering",
#             "hyperparameters": {"n_clusters": 3, "random_state": 42, "n_init": 10},
#             "reasoning": "K-Means is a good starting point for clustering analysis"
#         }
#     }

#     def __init__(self):
#         """
#         Initialize configuration generator
#         """
#         self.logger = logging.getLogger(self.__class__.__name__)

#     def analyze_dataset_characteristics(self, dataset_info: DatasetInfo) -> Dict[str, Any]:
#         """
#         Analyze dataset characteristics to determine best algorithm approach
#         Takes: DatasetInfo object
#         Returns: Dictionary with dataset characteristics
#         """
#         try:
#             characteristics = {}

#             # Dataset size categorization
#             if dataset_info.num_rows < 1000:
#                 characteristics["size_category"] = "small_dataset"
#             elif dataset_info.num_rows < 10000:
#                 characteristics["size_category"] = "medium_dataset"
#             else:
#                 characteristics["size_category"] = "large_dataset"

#             # Dimensionality analysis
#             if dataset_info.num_columns > 50:
#                 characteristics["dimensionality"] = "high_dimensional"
#             else:
#                 characteristics["dimensionality"] = "normal_dimensional"

#             # Data type analysis
#             numeric_columns = sum(1 for dtype in dataset_info.column_types.values()
#                                 if 'int' in dtype or 'float' in dtype)
#             categorical_columns = dataset_info.num_columns - numeric_columns

#             characteristics["numeric_ratio"] = numeric_columns / dataset_info.num_columns
#             characteristics["categorical_ratio"] = categorical_columns / dataset_info.num_columns

#             # Missing data analysis
#             characteristics["has_missing_data"] = dataset_info.has_missing_values

#             self.logger.info(f"Dataset characteristics analyzed: {characteristics}")
#             return characteristics

#         except Exception as e:
#             self.logger.error(f"Error analyzing dataset: {str(e)}")
#             raise

#     def generate_ml_config(self, dataset_info: DatasetInfo, characteristics: Dict[str, Any]) -> MLConfig:
#         """
#         Generate ML training configuration
#         Takes: DatasetInfo and dataset characteristics
#         Returns: MLConfig with recommended settings
#         """
#         try:
#             pipeline_type = dataset_info.pipeline_type

#             # Select algorithm based on pipeline type and characteristics
#             if pipeline_type == "Classification":
#                 algorithm_map = self.CLASSIFICATION_ALGORITHMS
#             elif pipeline_type == "Regression":
#                 algorithm_map = self.REGRESSION_ALGORITHMS
#             elif pipeline_type == "Clustering":
#                 algorithm_map = self.CLUSTERING_ALGORITHMS
#             else:
#                 raise ValueError(f"Unknown pipeline type: {pipeline_type}")

#             # Choose best algorithm based on characteristics
#             if pipeline_type == "Clustering":
#                 config = algorithm_map["default"]
#                 # Adjust number of clusters based on dataset size
#                 if dataset_info.num_rows > 1000:
#                     config["hyperparameters"]["n_clusters"] = min(10, max(3, dataset_info.num_rows // 100))
#             else:
#                 # Priority order for algorithm selection
#                 if characteristics["dimensionality"] == "high_dimensional":
#                     config = algorithm_map.get("high_dimensional", algorithm_map["medium_dataset"])
#                 else:
#                     config = algorithm_map[characteristics["size_category"]]

#                 # Special case for regression with many features
#                 if (pipeline_type == "Regression" and
#                     dataset_info.num_columns > 20 and
#                     characteristics["size_category"] == "small_dataset"):
#                     config = algorithm_map["regularization_needed"]

#             # Create ML configuration
#             ml_config = MLConfig(
#                 algorithm=config["algorithm"],
#                 hyperparameters=config["hyperparameters"],
#                 reasoning=config["reasoning"]
#             )

#             self.logger.info(f"Generated ML config: {ml_config.algorithm}")
#             return ml_config

#         except Exception as e:
#             self.logger.error(f"Error generating ML config: {str(e)}")
#             raise

#     def generate_evaluation_config(self, dataset_info: DatasetInfo) -> EvaluationConfig:
#         """
#         Generate evaluation configuration
#         Takes: DatasetInfo
#         Returns: EvaluationConfig with recommended evaluation settings
#         """
#         try:
#             pipeline_type = dataset_info.pipeline_type

#             # Select metrics based on pipeline type
#             if pipeline_type == "Classification":
#                 metrics = ["accuracy", "precision", "recall", "f1"]
#                 if len(set(dataset_info.column_types.get(dataset_info.target_column, []))) == 2:
#                     metrics.append("roc_auc")
#             elif pipeline_type == "Regression":
#                 metrics = ["mae", "mse", "rmse", "r2"]
#             elif pipeline_type == "Clustering":
#                 metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
#             else:
#                 metrics = []

#             # Adjust cross-validation based on dataset size
#             cv_folds = 5
#             if dataset_info.num_rows < 500:
#                 cv_folds = 3
#             elif dataset_info.num_rows > 10000:
#                 cv_folds = 10

#             evaluation_config = EvaluationConfig(
#                 metrics=metrics,
#                 cv_folds=cv_folds
#             )

#             self.logger.info(f"Generated evaluation config with metrics: {metrics}")
#             return evaluation_config

#         except Exception as e:
#             self.logger.error(f"Error generating evaluation config: {str(e)}")
#             raise

#     def generate_preprocessing_suggestions(self, dataset_info: DatasetInfo, characteristics: Dict[str, Any]) -> List[str]:
#         """
#         Generate preprocessing suggestions
#         Takes: DatasetInfo and characteristics
#         Returns: List of preprocessing suggestions
#         """
#         suggestions = []

#         try:
#             # Missing data suggestions
#             if dataset_info.has_missing_values:
#                 suggestions.append("Handle missing values before training (imputation or removal)")

#             # Categorical data suggestions
#             if characteristics["categorical_ratio"] > 0.3:
#                 suggestions.append("Consider encoding categorical variables (one-hot or label encoding)")

#             # Feature scaling suggestions
#             if characteristics["numeric_ratio"] > 0.5:
#                 suggestions.append("Consider feature scaling (StandardScaler or MinMaxScaler)")

#             # High dimensionality suggestions
#             if characteristics["dimensionality"] == "high_dimensional":
#                 suggestions.append("Consider dimensionality reduction (PCA or feature selection)")

#             # Small dataset suggestions
#             if characteristics["size_category"] == "small_dataset":
#                 suggestions.append("Consider cross-validation for robust model evaluation")
#                 suggestions.append("Watch for overfitting due to small dataset size")

#             self.logger.info(f"Generated {len(suggestions)} preprocessing suggestions")
#             return suggestions

#         except Exception as e:
#             self.logger.error(f"Error generating preprocessing suggestions: {str(e)}")
#             return []

#     def generate_warnings(self, dataset_info: DatasetInfo, characteristics: Dict[str, Any]) -> List[str]:
#         """
#         Generate warnings based on dataset characteristics
#         Takes: DatasetInfo and characteristics
#         Returns: List of warnings
#         """
#         warnings = []

#         try:
#             # Small dataset warnings
#             if dataset_info.num_rows < 100:
#                 warnings.append("Very small dataset - results may not be reliable")

#             # Imbalanced features warning
#             if dataset_info.num_columns > dataset_info.num_rows:
#                 warnings.append("More features than samples - high risk of overfitting")

#             # Missing target warning
#             if dataset_info.pipeline_type != "Clustering" and not dataset_info.target_column:
#                 warnings.append("No target column specified for supervised learning")

#             # High missing data warning
#             if dataset_info.has_missing_values:
#                 warnings.append("Dataset contains missing values - preprocessing required")

#             self.logger.info(f"Generated {len(warnings)} warnings")
#             return warnings

#         except Exception as e:
#             self.logger.error(f"Error generating warnings: {str(e)}")
#             return []

#     def generate_config(self, request: ConfigRequest) -> ConfigResponse:
#         """
#         Generate complete configuration
#         Takes: ConfigRequest
#         Returns: ConfigResponse with all configurations
#         """
#         try:
#             self.logger.info(f"Generating configuration for project: {request.project_name}")

#             # Analyze dataset
#             characteristics = self.analyze_dataset_characteristics(request.dataset_info)

#             # Generate configurations
#             ml_config = self.generate_ml_config(request.dataset_info, characteristics)
#             evaluation_config = self.generate_evaluation_config(request.dataset_info)

#             # Generate suggestions and warnings
#             preprocessing_suggestions = self.generate_preprocessing_suggestions(request.dataset_info, characteristics)
#             warnings = self.generate_warnings(request.dataset_info, characteristics)

#             response = ConfigResponse(
#                 ml_config=ml_config,
#                 evaluation_config=evaluation_config,
#                 preprocessing_suggestions=preprocessing_suggestions,
#                 warnings=warnings
#             )

#             self.logger.info("Configuration generation completed successfully")
#             return response

#         except Exception as e:
#             self.logger.error(f"Error generating configuration: {str(e)}")
#             raise

# # Initialize generator
# config_generator = ConfigGenerator()


# # @app.post("/generate-config", response_model=ConfigResponse)
# # async def generate_config(request: ConfigRequest):
# #     """
# #     Generate ML and evaluation configurations
# #     Takes: ConfigRequest with dataset info and preferences
# #     Returns: ConfigResponse with recommended configurations
# #     """
# #     try:
# #         logger.info(f"Received config request for project: {request.project_name}")

# #         # Validate request
# #         if not request.dataset_info.column_names:
# #             raise HTTPException(status_code=400, detail="Column names are required")

# #         if request.dataset_info.pipeline_type not in ["Classification", "Regression", "Clustering"]:
# #             raise HTTPException(status_code=400, detail="Invalid pipeline type")

# #         # Generate configuration
# #         response = config_generator.generate_config(request)

# #         logger.info(f"Configuration generated successfully for {request.project_name}")
# #         return response

# #     except Exception as e:
# #         logger.error(f"Error in generate_config endpoint: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint
#     Returns: Health status
#     """
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# @app.get("/algorithms/{pipeline_type}")
# async def get_available_algorithms(pipeline_type: str):
#     """
#     Get available algorithms for a pipeline type
#     Takes: pipeline_type
#     Returns: List of available algorithms
#     """
#     try:
#         if pipeline_type == "Classification":
#             algorithms = list(config_generator.CLASSIFICATION_ALGORITHMS.keys())
#         elif pipeline_type == "Regression":
#             algorithms = list(config_generator.REGRESSION_ALGORITHMS.keys())
#         elif pipeline_type == "Clustering":
#             algorithms = list(config_generator.CLUSTERING_ALGORITHMS.keys())
#         else:
#             raise HTTPException(status_code=400, detail="Invalid pipeline type")

#         return {"pipeline_type": pipeline_type, "algorithms": algorithms}

#     except Exception as e:
#         logger.error(f"Error getting algorithms: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # =======================================================================================================

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=7860)


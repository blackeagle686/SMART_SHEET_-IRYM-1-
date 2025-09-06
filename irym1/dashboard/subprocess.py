import subprocess
import tempfile
import re, os
from functools import wraps
import sys
import resource  # لتحديد استهلاك الموارد
import textwrap
from string import Template
import ast
from smartsheet_sync.fast_apis_side import get_generated_code
import json 
from bs4 import BeautifulSoup
from core_performance import performance_logger
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, TimeoutError

executor = ProcessPoolExecutor(max_workers=2)

def _execute_code_in_worker(code: str, data_path: str) -> dict:
    """
    Helper function that actually runs user code in a subprocess worker.
    """
    import subprocess, tempfile, sys

    # نكتب الكود في ملف مؤقت
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_file.flush()
        tmp_path = tmp_file.name

    try:
        # نشغل الكود
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Execution timed out", "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False}

def show_prcoesses(func):
    def wrapper(*args, **kwargs):
        print("\n==================== \n")
        print(f"current process: {func.__name__}")
        print(f"process inputs:")
        # for k, v in kwargs.items():
        #     print(f"{k} : {v}")
        result = func(*args, **kwargs)
        # if func.__name__ != "run_plotly_subprocess":
        #     print(f"results: {result}")
        return result
    return wrapper

def repeat(times: int):
    """Decorator to retry a function multiple times until it succeeds."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(times):
                try:
                    print(f"[Attempt {i+1}/{times}] Running function...")
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[Attempt {i+1}] Failed with: {e}")
                    last_exception = e
            # لو خلصت كل المحاولات ولسه في Exception
            raise last_exception
        return wrapper
    return decorator

allowed_libs = {
    "math", "statistics", "random",
    "numpy", "scipy", "pandas", "polars", "datatable",
    "matplotlib", "seaborn", "plotly", "bokeh", "altair",
    "sklearn", "xgboost", "lightgbm", "catboost",
    "tensorflow", "keras", "torch", "torchvision", "torchaudio", "pytorch_lightning", "jax",
    "transformers", "sentence_transformers", "spacy", "gensim", "nltk"
}
dangerous_calls = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help"
}

# ----------------------
# Static analysis
# ----------------------
def analyze_user_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error in user code: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_libs:
                    return False, f"Blocked import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in allowed_libs:
                return False, f"Blocked import: {node.module}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in dangerous_calls:
                return False, f"Blocked dangerous function: {node.func.id}"
    return True, "Code is safe"


#parallel
@repeat(5)
@performance_logger
@show_prcoesses
def run_in_subprocess(code: str, time_out: int = 60) -> dict:
    """
    Executes Python code in a subprocess with timeout.
    Captures:
      - DataFrame tables as HTML
      - Matplotlib/Seaborn plots as <img>
      - stdout/stderr text
    """

    is_safe, msg = analyze_user_code(code)
    if not is_safe:
        return {
            "success": False,
            "stdout": "<BLOCKED>",
            "stderr": msg,
            "raw_code": code,
        }

    # Inject helper hooks
    code = r"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import sys

plt.switch_backend("Agg")


def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{b64}'/>"


# Patch plt.show
import builtins

_old_show = plt.show


def _patched_show(*args, **kwargs):
    fig = plt.gcf()
    if fig:
        print(fig_to_html(fig))
    _old_show(*args, **kwargs)


plt.show = _patched_show


# Patch DataFrame display
_old_print = builtins.print


def _patched_print(*args, **kwargs):
    new_args = []
    for a in args:
        if isinstance(a, pd.DataFrame):
            new_args.append(a.to_html(classes="table table-bordered pandas-table"))
        else:
            new_args.append(a)
    _old_print(*new_args, **kwargs)


builtins.print = _patched_print
    """ + "\n" + code

    import subprocess, tempfile, os
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        result = subprocess.run(
            ["python3", temp_file],
            capture_output=True,
            text=True,
            timeout=time_out
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "raw_code": code
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timeout after {time_out} seconds",
            "raw_code": code
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)



#parallel
@performance_logger
@performance_logger
def process_code_prompt(prompt: str, data_path: str, columns: list) -> dict:
    """
    Generate Python code from prompt, run it in subprocess,
    and dynamically extract HTML tables, charts, insights.
    """
    full_prompt = "\n".join([
        "You are an expert Python data analyst.",
        f"Dataset path: {data_path}",
        "The dataframe must be named 'df'.",
        "Columns available (ONLY these allowed):",
        str(columns),
        "You must import required libraries.",
        "Final Requirement:",
        "Always print DataFrame previews with df.head().to_html() or any visualization outputs.",
        "Now, based on this dataset, generate Python code to:",
        prompt
    ])

    # Call LLM to generate code
    code_result = get_generated_code(full_prompt).get("code")

    # Run code in subprocess
    result = run_in_subprocess(code_result, time_out=60)

    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""

    # Extract HTML blocks dynamically
    html_blocks = re.findall(
        r"(<(?:table|h1|h2|h3|p)[^>]*>.*?</(?:table|h1|h2|h3|p)>|<img[^>]+?>)",
        stdout,
        flags=re.DOTALL | re.IGNORECASE
    )

    code_outputs = []
    for block in html_blocks:
        soup = BeautifulSoup(block, "html.parser")

        # Style tables
        for table in soup.find_all("table"):
            table["class"] = table.get("class", []) + ["pandas-table", "table-bordered"]

        # Style images
        for img in soup.find_all("img"):
            img["style"] = "max-width:600px; margin:10px; border:1px solid #ddd; border-radius:8px;"

        # Style paragraphs
        for p in soup.find_all("p"):
            p["class"] = p.get("class", []) + ["text-muted", "insight"]

        code_outputs.append(str(soup))

    return {
        "code_output": code_outputs,
        "code_error": stderr.strip(),
        "success": result.get("success"),
        "raw_code": code_result,
    }



# parallel
@show_prcoesses
@performance_logger
def run_plotly_subprocess(code, data_path: str, time_out: int = 60,
                          max_cols: int = 1, sample_size: int = 100,
                          instracution: dict = None) -> dict:
    """
    Executes Python code in a subprocess with timeout,
    generates Plotly Express charts as JSON (in parallel),
    wrapped between <<<PLOTLY_START>>> and <<<PLOTLY_END>>>.
    """

    # ✅ serialize مرة واحدة برا
    instr_json = json.dumps(instracution or {})

    plotly_code = f"""
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load dataset with sampling
df = pd.read_csv(r"{data_path}")
if len(df) > {sample_size}:
    df = df.sample(n={sample_size}, random_state=42)


def render_chart(kind, args):
    try:
        if kind == "hist":
            fig = px.histogram(df, x=args['col'], title=f"Histogram of {{args['col']}}",
                               color_discrete_sequence=args['colors'])
        elif kind == "box":
            fig = px.box(df, y=args['col'], title=f"Boxplot of {{args['col']}}",
                         color_discrete_sequence=args['colors'])
        elif kind == "violin":
            fig = px.violin(df, y=args['col'], box=True, points="all",
                            title=f"Violin plot of {{args['col']}}",
                            color_discrete_sequence=args['colors'])
        elif kind == "kde":
            hist_data = [df[args['col']].dropna()]
            fig = ff.create_distplot(hist_data, [args['col']], show_hist=False, show_rug=False,
                                     colors=args.get('colors', [px.colors.sequential.Inferno[3]]))
            fig.update_layout(title=f"KDE Density Plot of {{args['col']}}")
        elif kind == "heatmap":
            corr = df[args['cols']].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                            color_continuous_scale="Viridis")
        elif kind == "pie":
            top_values = df[args['col']].value_counts().head(15).index
            filtered_df = df[df[args['col']].isin(top_values)]
            fig = px.pie(filtered_df, names=args['col'],
                         title=f"Pie Chart of {{args['col']}}",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        else:
            return None
        return fig.to_json()
    except Exception as e:
        return json.dumps({{"error": str(e), "kind": kind, "col": args.get('col')}})


# ---- Handle direct instruction from user ----
try:
    instr = json.loads(r'''{instr_json}''')

    kind = instr.get("kind")
    col = instr.get("col")
    colors = instr.get("colors")

    if kind and col:
        if not colors or isinstance(colors, str):
            colors = px.colors.sequential.Viridis

        result = render_chart(kind=kind, args={{"col": col, "colors": colors}})
        if result:
            print("<<<PLOTLY_START>>>")
            print(result)
            print("<<<PLOTLY_END>>>")

        import sys
        sys.exit(0)

except Exception as e:
    print("<<<PLOTLY_START>>>")
    print(json.dumps({{"error": str(e)}}))
    print("<<<PLOTLY_END>>>")
    import sys
    sys.exit(0)


# ---- Auto-generate multiple charts in parallel ----
try:
    num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()[:{max_cols}]
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()[:{max_cols}]

    color_maps = ["Viridis", "Cividis", "Plasma", "Inferno", "Magma"]

    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, col in enumerate(num_cols):
            cmap = px.colors.sequential.__dict__[color_maps[i % len(color_maps)]]
            tasks.append(executor.submit(render_chart, "hist", {{"col": col, "colors": cmap}}))
            tasks.append(executor.submit(render_chart, "box",  {{"col": col, "colors": cmap}}))
            tasks.append(executor.submit(render_chart, "violin", {{"col": col, "colors": cmap}}))
            tasks.append(executor.submit(render_chart, "kde",  {{"col": col, "colors": [cmap[3]]}}))

        if 2 <= len(num_cols) <= 20:
            tasks.append(executor.submit(render_chart, "heatmap", {{"cols": num_cols}}))

        for col in cat_cols:
            if df[col].nunique() <= 15:
                tasks.append(executor.submit(render_chart, "pie", {{"col": col}}))

        for future in as_completed(tasks):
            result = future.result()
            if result:
                print("<<<PLOTLY_START>>>")
                print(result)
                print("<<<PLOTLY_END>>>")

except Exception as e:
    print("<<<PLOTLY_START>>>")
    print(json.dumps({{"error": str(e)}}))
    print("<<<PLOTLY_END>>>")
"""

    full_code = code + "\n" + plotly_code

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full_code)
            f.flush()
            temp_file = f.name

        result = subprocess.run(
            ["python3", temp_file],
            capture_output=True,
            text=True,
            timeout=time_out
        )
        print(result.stderr)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timeout after {time_out} seconds"
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


@performance_logger
def process_plotly_prompt(prompt: str, data_path: str, columns: list, instracution: dict = None, cols_ammont:int=5, sample:int=1000) -> dict:
    """
    Generate plotly charts (either by user instruction or auto).
    Returns dict with keys: 
        'plotly_output' (list of JSON plotly charts),
        'code_error',
        'success'
    """

    # ✅ لو مفيش instructions خليه {}
    if instracution is None:
        instracution = {}

    # مفيش كود LLM دلوقتي (بتشغل auto + instruction handling)
    code_result = ""  

    # ✅ مرر الـ instruction للـ subprocess
    result = run_plotly_subprocess(
        code_result,
        data_path,
        instracution=instracution, 
        max_cols=cols_ammont,
        sample_size=sample
    )

    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    # ✅ استخراج JSON blocks من stdout
    plotly_blocks = re.findall(
        r"<<<PLOTLY_START>>>(.*?)<<<PLOTLY_END>>>",
        stdout,
        flags=re.DOTALL
    )

    plotly_outputs = []
    for block in plotly_blocks:
        block = block.strip()
        try:
            plot_json = json.loads(block)  # parse JSON safely
            plotly_outputs.append(plot_json)
        except json.JSONDecodeError:
            plotly_outputs.append({
                "error": "Invalid JSON",
                "raw": block
            })
    return {
        "plotly_output": plotly_outputs,  # list of JSON objects
        "code_error": stderr.strip(),
        "success": result.get("success", False)
    }


@show_prcoesses
@performance_logger
def run_user_code_subprocess_secure(user_code: str, data_path: str, time_out: int = 60) -> dict:
    """
    Executes user Python code in a subprocess with:
      - Static analysis (block unsafe imports)
      - Runs user code dynamically (captures print outputs)
      - Generates automatic DataFrame reports (tables + charts in parallel)
    """

    import textwrap, tempfile, subprocess, sys, os, resource
    from string import Template

    # --- Security check ---
    is_safe, msg = analyze_user_code(user_code)
    if not is_safe:
        return {
            "success": False,
            "return_code": -2,
            "code_output": "<BLOCKED>",
            "code_error": msg,
            "raw_code": user_code,
        }

    def sanitize_user_code(user_code: str) -> str:
        code = textwrap.dedent(user_code).replace("\t", "    ").strip("\n")
        return code

    indented_user_code = sanitize_user_code(user_code)

    # ================= Execution Template =================
    user_code_template = Template(r"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, base64, io, traceback, sys, types
from concurrent.futures import ThreadPoolExecutor, as_completed

data_path = r"$data_path"
df = pd.read_csv(data_path)

# ------------------ Helpers ------------------
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{b64}'/>"

# --- Patch print() to handle DataFrames dynamically ---
_builtin_print = print
def safe_print(*args, **kwargs):
    out = []
    for a in args:
        try:
            if isinstance(a, pd.DataFrame):
                out.append(a.to_html())
            else:
                out.append(str(a))
        except Exception as e:
            out.append(f"[unprintable: {e}]")
    _builtin_print("\\n<<<user_code_output>>>")
    _builtin_print(" ".join(out), **kwargs)
    _builtin_print("<<<user_code_output>>>\\n")
print = safe_print

# --- Patch plt.show() to embed images ---
_old_show = plt.show
def patched_show(*args, **kwargs):
    fig = plt.gcf()
    if fig:
        print(fig_to_html(fig))
    _old_show(*args, **kwargs)
plt.show = patched_show

# ------------------ Auto Inspection Tasks ------------------
def gen_shape(df): 
    return f"<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} cols <br><br>"

def gen_head_tail(df):
    return "<h3>Head</h3>"+df.head().to_html()+"<h3>Tail</h3>"+df.tail().to_html()

def gen_info(df):
    info = pd.DataFrame({
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean()*100).round(2),
        "Unique Values": df.nunique(),
        "Dtype": df.dtypes
    })
    return "<h3>Column Info</h3>"+info.to_html()

def gen_describe(df):
    if df.select_dtypes(include="number").empty: return ""
    return "<h3>Numeric Summary</h3>"+df.describe().to_html()

def gen_corr(df):
    if df.select_dtypes(include="number").empty: return ""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    return "<h3>Correlation Matrix</h3>"+corr.to_html()+fig_to_html(fig)

# ------------------ Inspector ------------------
def inspect_dataframe(df, name="DataFrame"):
    try:
        print(f"<h2>📌 Report for {name}</h2>")
        tasks = [gen_shape, gen_head_tail, gen_info, gen_describe, gen_corr]
        out = []
        with ThreadPoolExecutor() as ex:
            fut_map = {ex.submit(fn, df): fn.__name__ for fn in tasks}
            for fut in as_completed(fut_map):
                try:
                    out.append(fut.result())
                except Exception as e:
                    out.append(f"<p style='color:red'>Error in {fut_map[fut]}: {e}</p>")
        print("".join(out))
    except Exception as e:
        print(f"<p style='color:red'>Error inspecting {name}: {e}</p>")

# ================= USER CODE =================
try:
$indented_user_code
except Exception as e:
    print(f"<p style='color:red'>Error in user code: {e}</p>")
    traceback.print_exc()

# ================= Final DF Inspection =================
try:
    if "df" not in locals():
        raise Exception("User code must define df")
    print("\\n<<<user_code_output>>>")
    inspect_dataframe(df, "Final df after user code")
    print("\\n<<<user_code_output>>>")
except Exception as e:
    print(f"<p style='color:red'>Error while generating report for df: {e}</p>")
    traceback.print_exc()
""")

    safe_code = user_code_template.substitute(
        data_path=data_path,
        indented_user_code=textwrap.indent(indented_user_code, "    ")
    )

    # ================= Execution =================
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(safe_code)
            f.flush()
            temp_file = f.name

        def limit_resources():
            try:
                resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 512*1024*1024))
                resource.setrlimit(resource.RLIMIT_CPU, (time_out, time_out))
            except Exception:
                pass

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=time_out,
            preexec_fn=limit_resources if sys.platform != "win32" else None
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "code_output": result.stdout if result.stdout else "<NO STDOUT>",
            "code_error": result.stderr if result.stderr else "<NO STDERR>",
            "raw_code": user_code
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "return_code": -1,
            "code_output": "<TIMEOUT>",
            "code_error": f"Timeout after {time_out} seconds",
            "raw_code": user_code
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


#parallel
@performance_logger
def process_user_code_secure(code: str, data_path: str) -> dict:
    """
    Runs user code securely and extracts HTML tables, headers, paragraphs, and images.
    Applies parallel processing on HTML block parsing for faster response.
    """
    result = run_user_code_subprocess_secure(code, data_path)

    stdout = result.get("code_output") or ""
    stderr = result.get("code_error") or ""

    # --- Extract user-defined blocks (between markers) ---
    user_blocks = re.findall(
        r"<<<user_code_output>>>(.*?)<<<user_code_output>>>",
        stdout,
        flags=re.DOTALL,
    )

    # --- Extract HTML fragments ---
    html_blocks = []
    for block in user_blocks:
        html_blocks.extend(
            re.findall(
                r"(<(?:table|h[1-6]|p|b|i|div|span|ul|ol|li)[^>]*?>.*?</(?:table|h[1-6]|p|b|i|div|span|ul|ol|li)>|<img[^>]+/?>)",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
        )

    # --- Block processor (styling tables & images) ---
    def process_block(block: str) -> str:
        soup = BeautifulSoup(block, "lxml")

        # Style tables
        for table in soup.find_all("table"):
            table_classes = set(table.get("class", []))
            table_classes.update(["pandas-table", "table", "table-bordered", "table-sm"])
            table["class"] = list(table_classes)

        # Style images
        for img in soup.find_all("img"):
            img_classes = set(img.get("class", []))
            img_classes.update(["img-fluid", "rounded", "shadow"])
            img["class"] = list(img_classes)

        return str(soup)

    # --- Parallelize block parsing ---
    with ThreadPoolExecutor() as executor:
        code_outputs = list(executor.map(process_block, html_blocks))

    return {
        "code_output": code_outputs,   # cleaned HTML blocks
        "user_code_out": user_blocks,  # raw blocks inside <<<user_code_output>>>
        "code_error": stderr.strip(),
        "success": bool(result.get("success")),
        "raw_code": code,
    }





#========================================== old but work ==========================================

# def run_in_subprocess(code: str, time_out: int = 60) -> dict:
#     """
#     Executes Python code in a subprocess with timeout.
#     Dynamically captures:
#       - Printed HTML/text
#       - DataFrames (to_html)
#       - Matplotlib/seaborn plots converted to <img>
#     """

#     is_safe, msg = analyze_user_code(code)
#     if not is_safe:
#         return {
#             "success": False,
#             "return_code": -2,
#             "stdout": "<BLOCKED>",
#             "stderr": msg,
#             "raw_code": code,
#         }

#     # Inject helper utilities into user code
#     code += r"""
# import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, base64, io, sys
# plt.switch_backend("Agg")

# def fig_to_html(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     b64 = base64.b64encode(buf.read()).decode("utf-8")
#     plt.close(fig)
#     return f"<img src='data:image/png;base64,{b64}'/>"

# # Auto-hook: whenever user creates plt.gcf(), auto print image
# import builtins
# _old_show = plt.show
# def _patched_show(*args, **kwargs):
#     fig = plt.gcf()
#     if fig:
#         print(fig_to_html(fig))
#     _old_show(*args, **kwargs)
# plt.show = _patched_show
# """

#     import subprocess, tempfile, os
#     temp_file = None
#     try:
#         with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
#             f.write(code)
#             f.flush()
#             temp_file = f.name

#         result = subprocess.run(
#             ["python3", temp_file],
#             capture_output=True,
#             text=True,
#             timeout=time_out
#         )

#         return {
#             "success": result.returncode == 0,
#             "stdout": result.stdout or "",
#             "stderr": result.stderr or "",
#             "raw_code": code
#         }

#     except subprocess.TimeoutExpired:
#         return {
#             "success": False,
#             "stdout": "",
#             "stderr": f"Timeout after {time_out} seconds",
#             "raw_code": code
#         }
#     finally:
#         if temp_file and os.path.exists(temp_file):
#             os.remove(temp_file)

# this not work effecianly but give good reports 
# parallel
# @show_prcoesses
# @performance_logger
# def run_user_code_subprocess_secure(user_code: str, data_path: str, time_out: int = 60) -> dict:
#     """
#     Executes user Python code in a subprocess with:
#       - Static analysis (block unsafe imports)
#       - Runs user code
#       - Generates automatic DataFrame reports (tables + charts in parallel)
#     """
    
#     is_safe, msg = analyze_user_code(user_code)
#     if not is_safe:
#         return {
#             "success": False,
#             "return_code": -2,
#             "code_output": "<BLOCKED>",
#             "code_error": msg,
#             "raw_code": user_code,
#         }

#     # ----------------------
#     # Execution template
#     # ----------------------
#     def sanitize_user_code(user_code: str) -> str:
#         # Remove common leading whitespace
#         code = textwrap.dedent(user_code)
#         # Replace tabs with 4 spaces
#         code = code.replace("\t", "    ")
#         # Strip accidental leading/trailing blank lines
#         code = code.strip("\n")
#         return code
    
#     indented_user_code = sanitize_user_code(user_code)
#     # safe to inject
    
#     user_code_template = Template(r"""
# import pandas as pd, numpy as np, matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt, seaborn as sns, base64, io, traceback, sys
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import types

# data_path = r"$data_path"
# df = pd.read_csv(data_path)

# def fig_to_html(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     b64 = base64.b64encode(buf.read()).decode("utf-8")
#     plt.close(fig)
#     return f"<img src='data:image/png;base64,{b64}'/>"

# # ------------------ Tasks ------------------
# def gen_shape(df): 
#     return f"<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} cols <br><br>"

# def gen_head_tail(df):
#     return "<h3>Head</h3>"+df.head().to_html()+"<h3>Tail</h3>"+df.tail().to_html()

# def gen_info(df):
#     info = pd.DataFrame({
#         "Non-Null Count": df.notnull().sum(),
#         "Missing Count": df.isnull().sum(),
#         "Missing %": (df.isnull().mean()*100).round(2),
#         "Unique Values": df.nunique(),
#         "Dtype": df.dtypes
#     })
#     return "<h3>Column Info</h3>"+info.to_html()

# def gen_describe(df):
#     if df.select_dtypes(include="number").empty: 
#         return ""
#     return "<h3>Numeric Summary</h3>"+df.describe().to_html()

# def gen_skew_kurt(df):
#     if df.select_dtypes(include="number").empty: 
#         return ""
#     stat = pd.DataFrame({
#         "Skewness": df.skew(numeric_only=True).round(2),
#         "Kurtosis": df.kurtosis(numeric_only=True).round(2)
#     })
#     return "<h3>Skewness & Kurtosis</h3>"+stat.to_html()

# def gen_corr(df):
#     if df.select_dtypes(include="number").empty: 
#         return ""
#     corr = df.corr(numeric_only=True)
#     fig, ax = plt.subplots(figsize=(8,6))
#     sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
#     return "<h3>Correlation Matrix</h3>"+corr.to_html()+fig_to_html(fig)

# def gen_value_counts(df):
#     out = ""
#     for col in df.select_dtypes(include="object"):
#         fig, ax = plt.subplots(figsize=(6,4))
#         vc = df[col].value_counts().head(10)
#         out += f"<h3>Top 10 values for {col}</h3>"+vc.to_frame().to_html()
#         vc.plot(kind="bar", ax=ax, title=f"{col} Top 10")
#         out += fig_to_html(fig)
#     return out

# def gen_outliers(df):
#     num_cols = df.select_dtypes(include="number")
#     if num_cols.empty: return ""
#     outlier_counts = {}
#     out = ""
#     for col in num_cols:
#         q1, q3 = num_cols[col].quantile([0.25, 0.75])
#         iqr = q3 - q1
#         lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
#         outliers = ((num_cols[col] < lower) | (num_cols[col] > upper)).sum()
#         outlier_counts[col] = outliers
#         fig, ax = plt.subplots(figsize=(5,3))
#         sns.boxplot(x=df[col], ax=ax)
#         ax.set_title(f"Boxplot - {col}")
#         out += fig_to_html(fig)
#     outlier_df = pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["Outlier Count"])
#     out += "<h3>Outliers (IQR)</h3>"+outlier_df.to_html()
#     return out

# def gen_key_insights(df):
#     out = "<h3>🔍 Key Insights</h3>"
#     if any(df.isnull().sum() > 0):
#         top_missing = df.isnull().sum().sort_values(ascending=False).head(1)
#         out += f"<p>⚠️ Most missing: <b>{top_missing.index[0]}</b> ({top_missing.iloc[0]})</p>"
#     constant_cols = [c for c in df.columns if df[c].nunique() == 1]
#     if constant_cols:
#         out += f"<p>⚠️ Constant cols: {constant_cols}</p>"
#     skewed = df.skew(numeric_only=True).sort_values(ascending=False)
#     if not skewed.empty:
#         out += f"<p>📈 Most skewed: <b>{skewed.index[0]}</b> ({skewed.iloc[0]:.2f})</p>"
#     high_card = [c for c in df.select_dtypes(include="object") if df[c].nunique() > 50]
#     if high_card:
#         out += f"<p>⚠️ High cardinality: {high_card}</p>"
#     return out

# # ------------------ Parallel Runner ------------------
# def inspect_dataframe(df, name="DataFrame"):
#     try:
#         print(f"<h2>📌 Report for {name}</h2>")
#         tasks = [gen_shape, gen_head_tail, gen_info, gen_describe, 
#                  gen_skew_kurt, gen_corr, gen_value_counts, gen_outliers, gen_key_insights]
#         out = []
#         with ThreadPoolExecutor() as ex:
#             fut_map = {ex.submit(fn, df): fn.__name__ for fn in tasks}
#             for fut in as_completed(fut_map):
#                 try:
#                     out.append(fut.result())
#                 except Exception as e:
#                     out.append(f"<p style='color:red'>Error in {fut_map[fut]}: {e}</p>")
#         print("".join(out))
#     except Exception as e:
#         print(f"<p style='color:red'>Error inspecting {name}: {e}</p>")

# # ===== USER CODE =====
# try:
#     _before_locals = set(locals().keys())
#     # Run injected code
# $indented_user_code

#     print("\n<<<user_code_output>>>")

#     _after_locals = dict(locals())

#     for name, value in _after_locals.items():
#         if name.startswith("__"): 
#             continue
#         if name in _before_locals:  
#             continue
#         if isinstance(value, (types.FunctionType, types.ModuleType, type)):
#             continue  # skip funcs, modules, classes
#         try:
#             print(f"{name} = {value}")
#         except Exception as e:
#             print(f"{name} (not printable: {e})")

#     print("<<<user_code_output>>>")

# except Exception as e:
#     print(f"<p style='color:red'>Error in user code: {e}</p>")
#     traceback.print_exc()

# # ===== Final DF Inspection =====
# try:
#     if "df" not in locals():
#         raise Exception("User code must define df")
#     print("\\n<<<user_code_output>>>")
#     inspect_dataframe(df, "Final df after user code")
#     print("\\n<<<user_code_output>>>")
# except Exception as e:
#     print(f"<p style='color:red'>Error while generating report for df: {e}</p>")
#     traceback.print_exc()
# """)

    
#     safe_code = user_code_template.substitute(
#         data_path=data_path,
#         indented_user_code=textwrap.indent(user_code, "    ")
#     )

#     # ----------------------
#     # Execution
#     # ----------------------
#     temp_file = None
#     try:
#         with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
#             f.write(safe_code)
#             f.flush()
#             temp_file = f.name

#         def limit_resources():
#             try:
#                 resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 512*1024*1024))
#                 resource.setrlimit(resource.RLIMIT_CPU, (time_out, time_out))
#             except Exception:
#                 pass

#         result = subprocess.run(
#             [sys.executable, temp_file],
#             capture_output=True,
#             text=True,
#             timeout=time_out,
#             preexec_fn=limit_resources if sys.platform != "win32" else None
#         )

#         return {
#             "success": result.returncode == 0,
#             "return_code": result.returncode,
#             "code_output": result.stdout if result.stdout else "<NO STDOUT>",
#             "code_error": result.stderr if result.stderr else "<NO STDERR>",
#             "raw_code": user_code
#         }

#     except subprocess.TimeoutExpired:
#         return {
#             "success": False,
#             "return_code": -1,
#             "code_output": "<TIMEOUT>",
#             "code_error": f"Timeout after {time_out} seconds",
#             "raw_code": user_code
#         }
#     finally:
#         if temp_file and os.path.exists(temp_file):
#             os.remove(temp_file)

# def process_code_prompt(prompt: str, data_path: str, columns: list) -> dict:
#     """
#     Generate code from prompt and run it in a persistent ProcessPoolExecutor worker.
#     Returns dict with keys: 'code_output', 'code_error', 'success'
#     """
#     full_prompt = "\n".join([
#         "You are an expert Python data analyst.",
#         "You will be given a dataset with a strict structure. Follow the rules carefully:",
#         f"Dataset path: {data_path}",
#         "Dataset type: csv",
#         "The variable name of the dataframe must be 'df'.",
#         "Available columns (you MUST use ONLY these, never invent extra ones):",
#         str(columns),
#         "You must import all required libraries and tools.",
#         "Instructions:",
#         "1. Load the dataset dynamically using the provided path.",
#         "2. Work only with the provided columns.",
#         "3. The code must be clean, modular, and well-structured.",
#         "4. Handle missing values, duplicates, and incorrect datatypes safely.",
#         "5. Detect column types dynamically instead of hardcoding them.",
#         "6. If an operation cannot be done with the given columns, explain it with a comment instead of inventing columns.",
#         "Final Requirement:",
#         "After cleaning the dataset, display a sample (first 5 rows) as an **HTML table** using `df.head().to_html()` and print it clearly.",
#         "Now, based strictly on this dataset, generate Python code to:",
#         prompt
#     ])

#     # استدعاء API لتوليد الكود
#     code_result = get_generated_code(full_prompt).get("code")

#     # نشغل الكود جوه ال worker بدل subprocess كل مرة
#     future = executor.submit(_execute_code_in_worker, code_result, data_path)
#     try:
#         result = future.result(timeout=20)
#     except TimeoutError:
#         return {"code_output": [], "code_error": "Execution timed out", "success": False, "raw_code": code_result}

#     stdout = result.get("stdout") or ""
#     stderr = result.get("stderr") or ""

#     # Extract tables, headers, images, and insights
#     html_blocks = re.findall(
#         r"(<(?:table|h1|h2|h3|p)[^>]*>.*?</(?:table|h1|h2|h3|p)>|<img[^>]+?>)",
#         stdout,
#         flags=re.DOTALL | re.IGNORECASE
#     )

#     code_outputs = []
#     for block in html_blocks:
#         soup = BeautifulSoup(block, "html.parser")

#         # Style tables
#         for table in soup.find_all("table"):
#             table["class"] = table.get("class", []) + ["pandas-table", "table-bordered"]

#         # Style images
#         for img in soup.find_all("img"):
#             img["style"] = "max-width:600px; margin:10px; border:1px solid #ddd; border-radius:8px;"

#         # Style paragraphs (insights)
#         for p in soup.find_all("p"):
#             p["class"] = p.get("class", []) + ["text-muted", "insight"]

#         code_outputs.append(str(soup))

#     return {
#         "code_output": code_outputs,
#         "code_error": stderr.strip() if stderr else "",
#         "success": result.get("success"),
#         "raw_code": code_result,
#     }

# def run_in_subprocess(code: str, time_out: int = 60) -> dict:
#     """
#     Executes Python code in a subprocess with timeout and 
#     generates extended HTML reports & charts for any DataFrame.
#     Parallelized for report tables & chart generation.
#     """
#     is_safe, msg = analyze_user_code(code)
#     if not is_safe:
#         return {
#             "success": False,
#             "return_code": -2,
#             "code_output": "<BLOCKED>",
#             "code_error": msg,
#             "raw_code": code,
#         }

#     code += r"""
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import base64, io
# from concurrent.futures import ThreadPoolExecutor, as_completed

# plt.switch_backend("Agg")

# def fig_to_html(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     b64 = base64.b64encode(buf.read()).decode("utf-8")
#     plt.close(fig)
#     return f"<img src='data:image/png;base64,{b64}'/>"

# def inspect_dataframe(df, name="DataFrame"):
#     try:
#         print(f"<h2>📌 Report for {name}</h2>")
#         print(f"<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} cols <br><br>")
#         print("<h3>Head</h3>")
#         print(df.head().to_html())
#         print("<h3>Tail</h3>")
#         print(df.tail().to_html())

#         tasks = {}

#         with ThreadPoolExecutor() as executor:
#             # Column Info
#             tasks["info"] = executor.submit(lambda: pd.DataFrame({
#                 "Non-Null Count": df.notnull().sum(),
#                 "Missing Count": df.isnull().sum(),
#                 "Missing %": (df.isnull().mean()*100).round(2),
#                 "Unique Values": df.nunique(),
#                 "Dtype": df.dtypes
#             }).to_html())

#             # Numeric Summary
#             if not df.select_dtypes(include="number").empty:
#                 num_df = df.select_dtypes(include="number")
#                 tasks["numeric_summary"] = executor.submit(lambda: num_df.describe().to_html())
#                 tasks["skew_kurt"] = executor.submit(lambda: pd.DataFrame({
#                     "Skewness": num_df.skew(numeric_only=True).round(2),
#                     "Kurtosis": num_df.kurtosis(numeric_only=True).round(2)
#                 }).to_html())
#                 tasks["corr_table"] = executor.submit(lambda: num_df.corr(numeric_only=True).to_html())
#                 tasks["corr_plot"] = executor.submit(lambda: fig_to_html(
#                     sns.heatmap(num_df.corr(numeric_only=True), annot=False, cmap="coolwarm").get_figure()
#                 ))

#             # Categorical top values + plots
#             for col in df.select_dtypes(include="object"):
#                 def top_values(col=col):
#                     table = df[col].value_counts().head(10).to_frame().to_html()
#                     fig, ax = plt.subplots(figsize=(6,4))
#                     df[col].value_counts().head(10).plot(kind="bar", ax=ax, title=f"{col} Top 10")
#                     return table + fig_to_html(fig)
#                 tasks[f"top_{col}"] = executor.submit(top_values)

#             # Outliers + Boxplots
#             if not df.select_dtypes(include="number").empty:
#                 num_cols = df.select_dtypes(include="number")
#                 def outlier_job(col):
#                     q1, q3 = num_cols[col].quantile([0.25, 0.75])
#                     iqr = q3 - q1
#                     lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
#                     outliers = ((num_cols[col] < lower) | (num_cols[col] > upper)).sum()
#                     fig, ax = plt.subplots(figsize=(5,3))
#                     sns.boxplot(x=df[col], ax=ax)
#                     ax.set_title(f"Boxplot - {col}")
#                     return col, outliers, fig_to_html(fig)
#                 for col in num_cols:
#                     tasks[f"outlier_{col}"] = executor.submit(outlier_job, col)

#         # Collect results
#         for key, fut in tasks.items():
#             res = fut.result()
#             if key == "info":
#                 print("<h3>Column Info</h3>"); print(res)
#             elif key == "numeric_summary":
#                 print("<h3>Numeric Summary</h3>"); print(res)
#             elif key == "skew_kurt":
#                 print("<h3>Skewness & Kurtosis</h3>"); print(res)
#             elif key == "corr_table":
#                 print("<h3>Correlation Matrix</h3>"); print(res)
#             elif key == "corr_plot":
#                 print(res)
#             elif key.startswith("top_"):
#                 print(f"<h3>Top 10 values for {key.split('_',1)[1]}</h3>"); print(res)
#             elif key.startswith("outlier_"):
#                 col, out_count, fig_html = res
#                 print(fig_html)
#                 print(f"<p><b>{col}:</b> {out_count} outliers detected</p>")

#         # Key Insights
#         print("<h3>🔍 Key Insights</h3>")
#         if any(df.isnull().sum() > 0):
#             top_missing = df.isnull().sum().sort_values(ascending=False).head(1)
#             print(f"<p>⚠️ Column with most missing values: <b>{top_missing.index[0]}</b> ({top_missing.iloc[0]})</p>")
#         constant_cols = [c for c in df.columns if df[c].nunique() == 1]
#         if constant_cols:
#             print(f"<p>⚠️ Constant columns detected: {constant_cols}</p>")
#         skewed = df.skew(numeric_only=True).sort_values(ascending=False)
#         if not skewed.empty:
#             print(f"<p>📈 Most skewed column: <b>{skewed.index[0]}</b> (Skew={skewed.iloc[0]:.2f})</p>")
#         high_card = [c for c in df.select_dtypes(include="object") if df[c].nunique() > 50]
#         if high_card:
#             print(f"<p>⚠️ High cardinality categorical columns: {high_card}</p>")

#     except Exception as e:
#         print(f"<p style='color:red'>Error inspecting {name}: {e}</p>")

# for var, val in list(globals().items()):
#     if isinstance(val, pd.DataFrame):
#         inspect_dataframe(val, var)
# """

#     import subprocess, tempfile, os
#     temp_file = None
#     try:
#         with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
#             f.write(code)
#             f.flush()
#             temp_file = f.name

#         result = subprocess.run(
#             ["python3", temp_file],
#             capture_output=True,
#             text=True,
#             timeout=time_out
#         )

#         return {
#             "success": result.returncode == 0,
#             "stdout": result.stdout,
#             "stderr": result.stderr
#         }

#     except subprocess.TimeoutExpired as e:
#         return {
#             "success": False,
#             "stdout": e.stdout or "",
#             "stderr": f"Timeout after {time_out} seconds"
#         }
#     finally:
#         if temp_file and os.path.exists(temp_file):
#             os.remove(temp_file)

#========================================== old but work ==========================================

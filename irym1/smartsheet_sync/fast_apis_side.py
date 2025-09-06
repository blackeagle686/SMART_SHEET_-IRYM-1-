import json 
import requests 
from django.conf import settings
fapi_base_url = settings.NGKEY 
from sheet_models.data_collector import GetData
from core_performance import performance_logger

def get_summary_result(prompt, data_cols):
    payload = {"prompt": prompt, "cols": data_cols}
    try:
        resp = requests.post(f"{fapi_base_url}/summary_result/", json=payload, timeout=45)

        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "message": f"Error response: {resp.text}",
                "status": resp.status_code
            }

    except requests.exceptions.RequestException as e:
        return {"message": str(e), "status": 500}
    
def send_data_info(data_provider_object, prompt):
    payload = {"long_description": data_provider_object.long_description, "prompt":prompt}
    print(payload)
    try:
        resp = requests.post(f"{fapi_base_url}/get_data_info/", json=payload, timeout=100)

        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "message": f"Error response: {resp.text}",
                "status": resp.status_code
            }

    except requests.exceptions.RequestException as e:
        print(f"\n\nthis error from djagno send_data_info(): {e}\n\n")
        return {"message": str(e), "status": 500}

 
def get_data_config(data_id):
    payload = {"id":data_id}
    try:
        resp = requests.post(f"{fapi_base_url}/get_data_config/", json=payload, timeout=100)

        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "message": f"Error response: {resp.text}",
                "status": resp.status_code
            }

    except requests.exceptions.RequestException as e:
        print(f"\n\nthis error from djagno get_data_info(): {e}\n\n")
        return {"message": str(e), "status": 500}

def get_generated_code(prompt: str):
    payload = {"prompt": prompt}
    url = f"{fapi_base_url}/generate_code/"

    try:
        resp = requests.post(url, json=payload, timeout=100)
        resp.raise_for_status()  # هيرفع استثناء لو status_code مش 2xx

        # حاول ترجّع JSON
        return resp.json()

    except requests.exceptions.HTTPError as http_err:
        return {
            "message": f"HTTP error occurred: {http_err}",
            "status": resp.status_code if resp else 500
        }
    except requests.exceptions.RequestException as req_err:
        return {"message": f"Request exception: {req_err}", "status": 500}
    except ValueError as json_err:
        return {"message": f"Invalid JSON response: {json_err}", "status": 500}
    
    

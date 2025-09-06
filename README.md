# SmartSheet (IRYM 1)  
![IRYM Series Logo](/home/tlk/Documents/IRYM-1-for-github/INSTALATION AND FASTAPI CODE/irym-2-inh.png)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Conda](https://img.shields.io/badge/Conda-Environment-green)
![Django](https://img.shields.io/badge/Django-5.2-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Yes-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-green)

**Version:** 0.1 (MVP)  
**Release Date:** July 30, 2025  
**Author / Lead Developer:** Mohamed Alaa  

**Contributors to This Version:**  
- **Mohamed Alaa** – Lead Development | Project Design | Backend | LLM Integration  
- **Ahmed Ali** – Machine Learning & Training Pipelines  
- **Abd El-Rahman Kamal** – Machine Learning & Evaluation Pipelines  


**Note:** IRYM 1 is the second project in the **I Can Read Your Mind** AI project series. Ahmed Ali and Abd El-Rahman Kamal **actively participated in building this specific version**, contributing essential features and development efforts.

---

## Overview

**SmartSheet (IRYM 1)** is a web-based platform that allows users to **upload datasets** (CSV, Excel, JSON) and describe in **natural language prompts** the type of analysis they want. 

The system leverages **Large Language Models (LLMs)** such as QWEN, and Qwen Coder:

1. Interpret user prompts 
2. Generate Python code dynamically 
3. Execute the code in a secure environment 
4. Return results as text or visualizations 

This enables users to analyze data without writing code themselves, making data insights accessible to non-technical users.

---

## Features

- **File Upload**: CSV, Excel, JSON  
- **Natural Language Prompts**: Describe analysis tasks in plain English  
- **Automated Code Generation**: LLM generates Python scripts dynamically  
- **Secure Execution Environment**: Code runs safely to prevent security risks  
- **Visualization Support**: Simple charts, plots, and tables  
- **User Accounts**: Sign-up, login, and profile management  
- **API Integration**: FastAPI for seamless backend services  

---

## Getting Started

### Prerequisites

- Python 3.10+  
- Conda (recommended for environment management)  
- Git  

---

## SmartSheet (IRYM 1) - Setup Instructions

### 1. Download the Installation File
- Available for **Windows, Linux, and Mac**.

### 2. Deploy the FastAPI Backend
- Copy the FastAPI code to a **GPU-enabled server**.
- Add your **ngrok authentication token** to obtain a **public URL**.
- Copy the generated public URL and paste it into the **NG_KEY** field in the SmartSheet app interface.

### 3. Run the Django Server
- After configuring the NG_KEY, start the **Django development server** to launch the web platform:

```bash
git clone https://github.com/username/SMART_SHEET_-IRYM-1-.git
cd SMART_SHEET_-IRYM-1-
# Activate conda environment
conda activate irym_1
# Start Django server
python manage.py runserver


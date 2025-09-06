# SmartSheet (IRYM 1)  

**Version:** 0.1 (MVP)  
**Date:** 2025-07-30  
**Author:** Mohamed Alaa  
**Team:** I Can Read Your Mind (IRYM)  
  - Mohammed Alaa
  - Ahmed Ali
  - Abd El-Rahman Kamal
---

## Overview

**SmartSheet (IRYM 1)** is a web-based platform that allows users to **upload datasets** (CSV, Excel, SQL, JSON, XML) and describe in **natural language prompts** the type of analysis they want.  

The system leverages **Large Language Models (LLMs)** such as QWEN, Phi, and others to:

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
- After configuring the NG_KEY, start the **Django development server** to launch the web platform.

```bash
git clone https://github.com/username/SMART_SHEET_-IRYM-1-.git
cd SMART_SHEET_-IRYM-1-


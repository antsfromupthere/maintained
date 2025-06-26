# MAINTAINED: Tool-Augmented AI for Wireless Network Deployment

**MAINTAINED** (autonoMous Artificial INTelligence Agent for Wireless NEtwork Deployment) is a hybrid framework that replaces parametric LLMs with computational grounding. It orchestrates specialized tools for geographic reasoning, propagation modeling, and optimization to support efficient, hallucination-free wireless network planning.

---

## 🚀 Project Goals

This framework demonstrates how tool-based agents can outperform traditional LLMs like ChatGPT in real-world wireless network deployment tasks.

---

## 📦 Installation Guide

### 1. Clone the Repository
git clone https://github.com/antsfromupthere/maintained.git  
cd maintained

### 2. Set Up Python Environment with Poetry
Make sure you have Poetry installed: https://python-poetry.org/docs/#installation

poetry install  
poetry shell

---

## 🔧 Additional Dependencies

### 3. Install OR-Tools (Google Optimization)
Follow the official guide: https://developers.google.com/optimization/install  
Or install directly:

pip install ortools

### 4. Install pycraf (Radio Propagation Toolkit)
pip install pycraf  
GitHub: https://github.com/bwinkel/pycraf

### 5. Install Ollama and Pull Qwen 3B Model
Ollama is used to run Qwen models locally.

Install Ollama: https://ollama.com/

Then pull the model:

ollama pull qwen:3b

### 6. Install smolagents
For agent tool orchestration:

pip install smolagents  
Repo: https://github.com/huggingface/smolagents

---

## 🗺️ Data Requirement

Download the high-resolution population density map for Saudi Arabia from Meta (Facebook Data for Good):  
https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps

Place the downloaded file in the following location:

maintained/hrpdm/sau_general_2020.csv

---

## 🙏 Acknowledgements

This project integrates and builds upon the following tools:

- pycraf: https://github.com/bwinkel/pycraf  
- Google OR-Tools: https://developers.google.com/optimization  
- smolagents: https://github.com/huggingface/smolagents  
- Ollama: https://ollama.com/
- Meta: https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps

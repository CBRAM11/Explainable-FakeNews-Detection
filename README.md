# Explainable-FakeNews-Detection

## Project Description

This project develops a deep learning-based system for fake news detection with interpretable outputs. The goal is to classify news text as **REAL or FAKE** while providing explanation signals that help users understand **why** the model made a particular decision.

The system combines a baseline machine learning approach with a transformer-based model (DistilBERT) and integrates an explainability mechanism based on perturbation analysis.

---

## Purpose

The focus of this project is not only achieving high classification accuracy, but also improving **model transparency and interpretability**. The system highlights important words and provides reasoning to analyze whether the model relies on meaningful signals or dataset-specific patterns.

---

## Features

* Fake news classification (**REAL / FAKE**)
* Confidence score for predictions
* Important word extraction using perturbation analysis
* Human-readable explanation of model decisions
* Interactive Streamlit-based user interface

---

## Models Used

* **TF-IDF + Logistic Regression** (Baseline model)
* **DistilBERT** (Deep learning model)

---

## Repository Structure

```
data/        : raw or processed dataset files  
notebooks/   : training, testing, and evaluation notebooks  
src/         : helper scripts and data pipeline code  
ui/          : Streamlit interface code  
results/     : metrics, outputs, and logs  
docs/        : architecture diagrams and interface screenshots  
saved_model/ : trained DistilBERT model files  
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Run the Notebook

Open and execute:

```
notebooks/setup.ipynb
```

This notebook:

* loads and preprocesses the dataset
* trains the baseline and deep learning models
* evaluates performance
* generates sample predictions

---

### 2. Launch the Interface

From the project root directory:

```bash
streamlit run ui/app.py
```

The interface allows users to:

* input news text
* receive predictions (REAL / FAKE)
* view confidence scores
* understand important words and explanations

---

## Dataset Information

The project uses publicly available fake news datasets consisting of labeled news articles. Data preprocessing includes:

* removing null values
* cleaning and formatting text
* removing duplicate entries
* sampling for efficient training

---

## Current Results

* Logistic Regression Accuracy: **~97.85%**
* DistilBERT Accuracy: **~99.8%**
* High precision, recall, and F1-score

---

## Key Insight

Although the model achieves very high accuracy, explainability analysis reveals that it may rely on:

* stylistic patterns
* named entities (locations, organizations)
* writing structure

rather than true semantic understanding of misinformation.

---

## Known Limitations

* The model predicts based on **textual patterns**, not factual verification
* High accuracy may be influenced by dataset-specific shortcuts
* Explainability is approximate and may not fully capture true reasoning
* The system should not be used as a standalone fact-checking tool

---

## Deliverable 3 Improvements
- Improved UI layout and design
- Cleaner explanation output
- Added input validation
- Added low-confidence warning
- Better user experience

## Author

Chathur Buja Ram Bathi
Email: cbathi@ufl.edu

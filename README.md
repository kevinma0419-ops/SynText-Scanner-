# 🔍 SynText-Scanner: Investigating Data Drift & LLM Detection Limits

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-green)

## 📌 Executive Summary (The TL;DR)
This project began as a standard Natural Language Processing (NLP) classification task to distinguish between human-written and AI-generated text. However, it quickly evolved into a rigorous investigation of **Data Drift** and the **Statistical Turing Test**. 

Through a 4-phase experimental pipeline ranging from traditional TF-IDF + XGBoost models to deep semantic fine-tuning (BERT) and zero-shot perplexity analysis (GPT-2), this project conclusively demonstrates a critical industry reality: **State-of-the-Art (SOTA) generative AI models (circa 2025) have successfully bypassed traditional text detection mechanisms.** They mimic human linguistic variance and "burstiness" so perfectly that static detection models trained on historical data are now statistically obsolete.

## 🏗️ Project Architecture & Methodology

The investigation was conducted across four distinct phases to systematically eliminate confounding variables:

### Phase 1: EDA & Baseline Feature Engineering
* **Action:** Conducted exploratory data analysis on token lengths, lexical diversity, and syntax. Cleaned text using Lemmatization and stop-word removal.
* **Result:** Extracted a TF-IDF matrix to establish a baseline for lexical differences.

### Phase 2: Traditional Machine Learning (The Baseline)
* **Action:** Trained Logistic Regression and XGBoost classifiers on the TF-IDF features.
* **Result:** Accuracy stagnated at **~50%**. This proved that surface-level frequency metrics are insufficient to catch modern AI.

### Phase 3: Deep Learning & The "Domain Shift" Discovery
* **Action:** Fine-tuned `distilbert-base-uncased` and the full `bert-base-uncased` (110M parameters) using PyTorch and Hugging Face `Trainer`. 
* **Discovery:** The model achieved near-perfect memorization on the historical training set (Training Loss: 0.0002) but failed completely on the modern 2025 test set (Validation Loss: 4.51, Accuracy: ~50%).
* **Insight:** This Reverse Domain Shift Experiment mathematically proved that the semantic "fingerprints" present in older LLMs have been completely eradicated in modern generative models.

### Phase 4: Zero-Shot Perplexity (PPL) Analysis (The Final Proof)
* **Action:** Bypassed training entirely. Deployed a causal language model (`distilgpt2`) to calculate the raw cross-entropy loss and Perplexity (PPL) of both human and AI texts in a zero-shot environment.
* **Discovery:** The Kernel Density Estimate (KDE) plot revealed that the perplexity distributions of Human and AI-generated texts are almost **perfectly superimposed**. Modern AI has successfully replicated the "unpredictability" of human writing.

## 🚀 Key Takeaways for Data Science
This project serves as a "Perfect Negative Result." It rigorously proves that:
1. **Static post-hoc AI text detection is fundamentally broken.** 2. The boundary between synthetic and organic text has dissolved within linguistic feature spaces.
3. Future verification methods must rely on cryptographic watermarking (e.g., SynthID) or verifiable provenance tracking, rather than statistical linguistic analysis.

## 💻 Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **NLP Libraries:** Hugging Face `transformers`, `nltk`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`

## 📂 Repository Structure
```text
├── data/                   # Raw and processed datasets
├── notebooks/              
│   ├── 01_EDA_and_Cleaning.ipynb          # Data processing & TF-IDF
│   ├── 02_XGBoost_Baseline.ipynb          # Traditional ML models
│   ├── 03_Deep_Learning_Transformer.ipynb # BERT fine-tuning & evaluation
│   └── 04_Zero_Shot_Perplexity.ipynb      # GPT-2 PPL distribution analysis
├── .gitignore              # Ignoring heavy model checkpoints
└── README.md               # Project documentation
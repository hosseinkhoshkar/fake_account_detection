# Twitter Fake Account Detection – Comparative Study

## 📌 Overview
This project presents a **comparative study of machine learning, deep learning, and transformer-based models** for detecting fake accounts on Twitter.  
The framework systematically evaluates multiple algorithms, compares their performance, and analyzes why certain models outperform others.  

The implemented models include:
- **Classical Machine Learning:** K-Nearest Neighbors (KNN), Support Vector Machine (SVM with RBF kernel), Random Forest  
- **Boosting Methods:** XGBoost, LightGBM  
- **Deep Learning:** Autoencoder, Auxiliary Classifier GAN (ACGAN)  
- **Language Models:** BERT, LLaMA  

---

## 📂 Project Structure
```
twitter_fake_detection/
├─ main.py                  # Main entry point for running experiments
├─ requirements.txt         # Project dependencies
├─ README.md                # Project documentation

├─ data/
│  └─ final_dataset.csv     # Final preprocessed dataset

├─ algorithms/              # Implementations of algorithms
│  ├─ base.py
│  ├─ knn.py
│  ├─ svm.py
│  ├─ random_forest.py
│  ├─ xgb.py
│  ├─ lgbm.py
│  ├─ autoencoder.py
│  ├─ acgan.py
│  ├─ bert.py
│  └─ llama.py

├─ evaluation/
│  └─ evaluation.py         # Evaluation metrics and utilities

├─ results/                 # Experiment outputs
│  ├─ results_comparison.csv
│  ├─ f1_scores.png
│  ├─ accuracy_scores.png
│  └─ metrics_charts.pdf

├─ notebooks/
│  ├─ run_on_colab.ipynb    # Notebook for running experiments on Google Colab
│  ├─ run_on_colab.html
│  └─ process_tweets.py     # Tweet preprocessing script
```

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/username/twitter_fake_detection.git
cd twitter_fake_detection
pip install -r requirements.txt
```

---

## 🚀 Usage
Run experiments with:
```bash
python main.py
```

### Options
- Modify **main.py** to choose algorithms to run.  
- Adjust hyperparameters inside each file under `algorithms/`.  
- Results (Accuracy, Precision, Recall, F1, AUC) will be saved in the `results/` folder.  

---

## 📊 Evaluation Metrics
The following metrics are used to compare models:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC (where applicable)  

---

## 🏆 Results
Example evaluation summary:

| Algorithm      | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------------|----------|-----------|--------|-------|---------|
| KNN            | 97.67%   | 97.96%    | 97.50% | 97.73%|   -     |
| SVM            | 98.56%   | 99.21%    | 97.97% | 98.58%|   -     |
| Random Forest  | 99.04%   | 100%      | 98.12% | 99.05%|   -     |
| XGBoost        | 99.92%   | 100%      | 99.84% | 99.92%|   -     |
| LightGBM       | 100%     | 100%      | 100%   | 100%  |   -     |
| Autoencoder    | 51.36%   | 51.36%    | 100%   | 67.87%|   -     |
| ACGAN          | 97.75%   | 98.57%    | 97.03% | 97.80%|  0.9955 |
| BERT           | 81.38%   | 83.33%    | 79.69% | 81.47%|  0.8937 |
| LLaMA          | 50.72%   | 51.77%    | 59.38% | 55.31%|  0.4905 |

---

## 📖 Notes
- **LightGBM** achieved the best overall performance.  
- **Classical methods (KNN, SVM, RF)** remained competitive despite their simplicity.  
- **Deep learning models (Autoencoder, ACGAN)** showed mixed results, with ACGAN performing strongly.  
- **Language models (BERT, LLaMA)** provided valuable insights into text features but underperformed compared to boosting methods.  

---

## 👨‍🎓 Author
Developed as part of a Master’s thesis project at **GISMA University**.  

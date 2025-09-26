
# 🕵️ Twitter Fake Account Detection

This project is a system for detecting fake accounts on Twitter using various **Machine Learning** and **Deep Learning** algorithms.  
The goal is to compare multiple algorithms based on metrics such as `Accuracy`, `Precision`, `Recall`, `F1`, and `ROC-AUC`.

---

## 📂 Project Structure
```
twitter_fake_detection/
│
├── algorithms/                # Algorithms
│   ├── base.py                 # BaseAlgorithm class
│   ├── knn.py                  # KNNAlgorithm
│   ├── svm.py                  # SVMAlgorithm
│   ├── random_forest.py        # RandomForestAlgorithm
│   ├── xgboost.py              # XGBoostAlgorithm
│   ├── lightgbm.py             # LightGBMAlgorithm
│   ├── autoencoder.py          # AutoencoderAlgorithm
│   ├── gan.py                  # GANClassifierAlgorithm (ACGAN)
│   ├── bert.py                 # BertAlgorithm (DistilBERT multilingual)
│   └── llama.py                # LLaMAAlgorithm (Zero/Few-shot)
│
├── evaluation/
│   └── evaluation.py           # Evaluation function for comparing algorithms
│
├── data/                       # Dataset (train/test)
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test.csv
│   └── y_test.csv
│
├── main.py                     # Main entry point
├── requirements.txt            # Required dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Algorithms
The following algorithms are implemented in this project:
- **KNN** (K-Nearest Neighbors)  
- **SVM** (Support Vector Machine)  
- **Random Forest**  
- **XGBoost**  
- **LightGBM**  
- **Autoencoder** (for anomaly detection)  
- **GAN (ACGAN)**  
- **BERT (DistilBERT multilingual)**  
- **LLaMA (Zero/Few-shot)**  

---

## 🚀 How to Run on Google Colab
1. Upload the project as a ZIP file to your Google Drive.  
2. Open the `colab_setup.ipynb` file in Google Colab.  
3. Run the cells in order to extract the project, install dependencies, and execute `main.py`.  

---

## 📦 Local Installation (Optional)
If you want to run the project locally:
```bash
git clone https://github.com/username/twitter_fake_detection.git
cd twitter_fake_detection
pip install -r requirements.txt
python main.py
```

---

## ✍️ Author
- Hossein Khoshkar Raste Kenari

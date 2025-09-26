import pandas as pd
from sklearn.model_selection import train_test_split
from algorithms.knn import KNNAlgorithm
from algorithms.svm import SVMAlgorithm
from algorithms.random_forest import RandomForestAlgorithm
from algorithms.xgb import XGBAlgorithm
from algorithms.lgbm import LGBMAlgorithm
from algorithms.autoencoder import AutoencoderAlgorithm
from algorithms.acgan import ACGANAlgorithm
from algorithms.bert import BertAlgorithm
from algorithms.llama import LlamaAlgorithm
from evaluation.evaluation import plot_confusion_matrices
from evaluation.evaluation import (
    evaluate_algorithms,
    plot_f1_scores,
    plot_accuracy_scores,
    save_charts_to_pdf
)

df = pd.read_csv("data/final_dataset.csv")

# TEST_ROWS = 200
# if len(df) > TEST_ROWS:
#     df = df.sample(TEST_ROWS, random_state=42).reset_index(drop=True)

y = df["label"].values.ravel()
X = df.drop(columns=["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

algorithms = [
    KNNAlgorithm(),
    SVMAlgorithm(),
    RandomForestAlgorithm(),
    XGBAlgorithm(),
    LGBMAlgorithm(),
    AutoencoderAlgorithm(),
    ACGANAlgorithm(input_dim=X_train.shape[1]),
    BertAlgorithm(),
    LlamaAlgorithm()
]

for algo in algorithms:
    print(f"Training {algo.name} ...")
    algo.fit(X_train, y_train)

results_df = evaluate_algorithms(algorithms, X_test, y_test)
results_df.to_csv("results_comparison.csv", index=False)

plot_f1_scores(results_df, "f1_scores.png")
plot_accuracy_scores(results_df, "accuracy_scores.png")
save_charts_to_pdf(results_df, "metrics_charts.pdf")
plot_confusion_matrices(algorithms, X_test, y_test, pdf_filename="confusion_matrices.pdf", normalize=True)

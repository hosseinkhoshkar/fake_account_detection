import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_algorithms(algorithms, X_test, y_test):
    results = []
    for algo in algorithms:
        y_pred = algo.predict(X_test)
        y_proba = None
        if hasattr(algo, "predict_proba"):
            try:
                y_proba = algo.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        results.append({
            "Algorithm": getattr(algo, "name", algo.__class__.__name__),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC-AUC": round(roc, 4) if roc is not None else "N/A"
        })
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))
    return df

def plot_f1_scores(df, filename="f1_scores.png", show_console=True):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Algorithm"], df["F1"])
    plt.ylabel("F1 Score")
    plt.title("F1 Scores by Algorithm")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show_console:
        plt.show()
    plt.close()

def plot_accuracy_scores(df, filename="accuracy_scores.png", show_console=True):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Algorithm"], df["Accuracy"])
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Algorithm")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show_console:
        plt.show()
    plt.close()

def save_charts_to_pdf(df, filename="metrics_charts.pdf", show_console=True):
    with PdfPages(filename) as pdf:
        plt.figure(figsize=(10, 6))
        plt.bar(df["Algorithm"], df["F1"])
        plt.ylabel("F1 Score")
        plt.title("F1 Scores by Algorithm")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        pdf.savefig()
        if show_console:
            plt.show()
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.bar(df["Algorithm"], df["Accuracy"])
        plt.ylabel("Accuracy")
        plt.title("Accuracy by Algorithm")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        pdf.savefig()
        if show_console:
            plt.show()
        plt.close()

def plot_confusion_matrices(algorithms, X_test, y_test, pdf_filename="confusion_matrices.pdf", normalize=True):
    classes = np.unique(y_test)
    with PdfPages(pdf_filename) as pdf:
        for algo in algorithms:
            try:
                y_pred = algo.predict(X_test)
            except Exception:
                continue
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            if normalize and cm.sum(axis=1).all():
                cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - {getattr(algo, 'name', algo.__class__.__name__)}")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45, ha="right")
            plt.yticks(tick_marks, classes)
            fmt = ".2f" if normalize else "d"
            thresh = cm.max() / 2.0 if cm.size else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], fmt),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"Saved confusion matrices to {pdf_filename}")

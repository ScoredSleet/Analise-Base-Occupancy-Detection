import numpy as np
from sklearn.metrics import confusion_matrix

from typing import Any

def compute_metrics(y_true, y_pred, pos_label: int = 1) -> dict[str, Any]:
    """Calcula Accuracy, Precision, Sensitivity, Specificity, F1 e MH.

    Todas as métricas derivam da MESMA matriz de confusão, garantindo
    consistência interna. Fórmulas seguem as equações do TCC.

    Parametros:
        y_true, y_pred : array-like
        pos_label : int
            Classe considerada "positiva" no contexto do problema

    Returns
        Metricas
    """
    labels = [0, pos_label]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    # Acurácia
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # Precisão
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Sensitividade
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Especificidade
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1-Score
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0 else 0.0
    )

    # Média Harmônica
    mh = (
        2 * sensitivity * specificity / (sensitivity + specificity)
        if (sensitivity + specificity) > 0 else 0.0
    )

    return {
        "accuracy":    float(accuracy),
        "precision":   float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1":          float(f1),
        "mh":          float(mh),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
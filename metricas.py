import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline


from preprocessamento import carregar_base
from typing import Any

import os

'''
Tentarei usar como base para o random search o https://www.datacamp.com/pt/tutorial/parameter-optimization-machine-learning-models
'''


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    """Calcula Accuracy, Precision, Sensitivity, Specificity, F1 e MH.

    Todas as métricas derivam da MESMA matriz de confusão, garantindo
    consistência interna. Fórmulas seguem as equações do TCC.

    Parametros:
        y_true, y_pred : array-like

    Returns
        Metricas
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    return {
        "Acc":   accuracy_score(y_true, y_pred),
        "Prec":  precision_score(y_true, y_pred, zero_division=0),
        "Sens":  recall_score(y_true, y_pred, zero_division=0),
        "Espec": tn / (tn + fp) if tn + fp else 0.0,
        "F1":    f1_score(y_true, y_pred, zero_division=0),
        "cm":    cm,
    }
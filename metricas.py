import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.pipeline import Pipeline


from preprocessamento import carregar_base
from typing import Any

import os

'''
Tentarei usar como base para o random search o https://www.datacamp.com/pt/tutorial/parameter-optimization-machine-learning-models
'''


def compute_metrics(y_true, y_pred, pos_label: int = 1, ignore_cm = False) -> dict[str, Any]:
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
    
    if ignore_cm == True:
        return {
            "accuracy":    float(accuracy),
            "precision":   float(precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1":          float(f1),
            #"mh":          float(mh),
        }

    return {
        "accuracy":    float(accuracy),
        "precision":   float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1":          float(f1),
        #"mh":          float(mh),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def treino(
    models: dict,
    scalers: dict,
    PATH_treino: str,
    formatar: bool = True,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """
        Treina cada combinação (modelo, scaler) no datatraining completo.
    """
    X_train, y_train = carregar_base(PATH_treino, return_X_y=True, formatar=formatar)

    resultados_treino = {}
    modelos_treinados = {}

    for nome_modelo, modelo_base in models.items():
        for nome_scaler, scaler in scalers.items():
            chave = f"{nome_modelo}_{nome_scaler}"

            # monta pipeline: scaler (ou passthrough) + modelo
            scaler_obj = scaler if scaler is not None else "passthrough"
            if not isinstance(scaler_obj, str):
                scaler_obj = clone(scaler_obj)

            pipe = Pipeline([
                ("scaler", scaler_obj),
                ("model",  clone(modelo_base)),
            ])
            pipe.fit(X_train, y_train)

            modelos_treinados[chave] = {"pipeline": pipe}

            m = compute_metrics(y_train, pipe.predict(X_train))
            resultados_treino[chave] = m

            if verbose:
                print(f"[Treino] {chave:<15} -> "
                      f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}")

    return resultados_treino, modelos_treinados

def validacao(
    modelos_treinados: dict,
    PATHs_teste: list[str],
    formatar: bool = True,
    verbose: bool = False,
) -> dict:
    """Aplica os pipelines já treinados nos datasets de teste."""
    
    dados_teste = [carregar_base(p, return_X_y=True, formatar=formatar) for p in PATHs_teste]
    nomes_teste = [os.path.splitext(os.path.basename(p))[0] for p in PATHs_teste]

    resultados_teste = {}

    for chave, artefatos in modelos_treinados.items():
        pipe = artefatos["pipeline"]
        resultados_teste[chave] = {}

        for (X_t, y_t), nome_teste in zip(dados_teste, nomes_teste):

            m = compute_metrics(y_t, pipe.predict(X_t))
            for metrica_nome, valor in m.items():
                resultados_teste[chave][f"{nome_teste}_{metrica_nome}"] = valor

            if verbose:
                print(f"[{nome_teste}] {chave:<15}: f1={m['f1']:.4f}  acc={m['accuracy']:.4f}")

    return resultados_teste
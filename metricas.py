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
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from preprocessamento import carregar_base
from typing import Any

import os

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
            "mh":          float(mh),
        }

    return {
        "accuracy":    float(accuracy),
        "precision":   float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1":          float(f1),
        "mh":          float(mh),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    
def calcular_metricas_fold(models, scalers, PATHs_treino, PATHs_teste, colunas, random_state = 42, n_folds = 5):
    dados_treino = [carregar_base(p, return_X_y=True) for p in PATHs_treino]
    dados_teste  = [carregar_base(p, return_X_y=True) for p in PATHs_teste]

    nome_de = lambda p: os.path.splitext(os.path.basename(p))[0]
    nomes_treino = [nome_de(p) for p in PATHs_treino]
    nomes_teste  = [nome_de(p) for p in PATHs_teste]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    resultados_cv    = {}
    resultados_teste = {}

    for (X_train, y_train), nome_treino in zip(dados_treino, nomes_treino):
        for nome_modelo, modelo_base in models.items():
            for nome_scaler, scaler in scalers.items():
                chave = f"{nome_treino}__{nome_modelo}_{nome_scaler}"

                # ---------- Fase 1: Cross Validation ----------
                metricas_folds = []
                for i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    X_tr  = X_train.iloc[tr_idx]
                    X_val = X_train.iloc[val_idx]
                    y_tr  = y_train.iloc[tr_idx]
                    y_val = y_train.iloc[val_idx]

                    X_tr_scaled, scaler_fitado = scaler(X_tr, colunas)
                    X_val_scaled = X_val.copy()
                    X_val_scaled[colunas] = scaler_fitado.transform(X_val[colunas])

                    model = clone(modelo_base)
                    model.fit(X_tr_scaled[colunas], y_tr)

                    y_pred = model.predict(X_val_scaled[colunas])
                    
                    m = compute_metrics(y_true=y_val, y_pred=y_pred)
                    metricas_folds.append(m)
                    
                    #print(f"  [{chave}] fold {i}: "
                          #f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}")

                resultados_cv[chave] = {
                    f"{k}_mean": float(np.mean([m[k] for m in metricas_folds]))
                    for k in metricas_folds[0]
                }
                
                resultados_cv[chave].update({
                    f"{k}_std": float(np.std([m[k] for m in metricas_folds]))
                    for k in metricas_folds[0]
                })

                print(f"[CV] {chave} → f1 = "
                      f"{resultados_cv[chave]['f1_mean']:.4f} "
                      f"± {resultados_cv[chave]['f1_std']:.4f}")

                # ---------- Fase 2: retreino no treino inteiro + testes ----------
                X_train_full_scaled, scaler_full = scaler(X_train, colunas)
                model_final = clone(modelo_base)
                model_final.fit(X_train_full_scaled[colunas], y_train)

                resultados_teste[chave] = {}
                for (X_t, y_t), nome_teste in zip(dados_teste, nomes_teste):
                    
                    X_t_scaled = X_t.copy()
                    X_t_scaled[colunas] = scaler_full.transform(X_t[colunas])

                    y_pred = model_final.predict(X_t_scaled[colunas])
                    m = compute_metrics(y_true=y_t, y_pred=y_pred)

                    resultados_teste[chave][f"{nome_teste}_f1"]       = float(m["f1"])
                    resultados_teste[chave][f"{nome_teste}_accuracy"] = float(m["accuracy"])

                    print(f"[{nome_teste}] {chave}: "
                          f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}")
                
                print('\n')

    return resultados_cv, resultados_teste
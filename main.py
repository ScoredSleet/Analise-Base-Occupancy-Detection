'''
Atribuições do projeto e base de Dados

- https://archive.ics.uci.edu/dataset/357/occupancy+detection?utm_source=chatgpt.com
    - Base de dados sobre detecção de ocupação
    - Artigo em anexo que elucida bem o que é
        Olhar EDA antes qualquer coisa
        
# Faixas válidas por coluna
# date          - (timestamp)
# Temperature   - [-40, 80] °C
# Humidity      - [0, 100] % RH
# Light         - [1, 40000] lux
# CO2           - [0, 2000] ppm
# HumidityRatio - derivada (sem faixa própria)
# Occupancy     - {0, 1}
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from preprocessamento import z_score, min_max
from metricas import calcular_metricas_fold

import numpy as np
import pandas as pd

from preprocessamento import carregar_base

COLUNAS = ['Temperature','Humidity','Light','CO2']

PATH_TREINO = "./dataset/occupancy_detection/dataset_csv/datatraining.csv" 
PATH_TEST = "./dataset/occupancy_detection/dataset_csv/datatest.csv" 
PATH_TEST2 = "./dataset/occupancy_detection/dataset_csv/datatest2.csv" 

RANDOM_STATE = 42

models = {
    "KNN": KNeighborsClassifier(), 
    "SVM": SVC(random_state=RANDOM_STATE)
}

scalers = {
    'Z-Score': z_score,
    'Min-Max': min_max
}

resultados_cv, resultados_teste = calcular_metricas_fold(models, scalers, [PATH_TREINO], [PATH_TEST, PATH_TEST2], COLUNAS, RANDOM_STATE, 5)

        
# =====================================================================
# DATAFRAMES
# =====================================================================
df_cv = pd.DataFrame(resultados_cv).T
df_cv.index.name = "modelo_scaler"
df_cv = df_cv.sort_values("f1_mean", ascending=False)

df_teste = pd.DataFrame(resultados_teste).T
df_teste.index.name = "modelo_scaler"
df_teste = df_teste.reindex(df_cv.index)

# combo + deltas
df_combo = df_cv.join(df_teste)
for col in df_teste.columns:
    if col.endswith("_f1"):
        nome = col[:-3]                                     # "datatest" / "datatest2"
        df_combo[f"delta_{nome}_f1"] = df_combo[col] - df_combo["f1_mean"]
df_combo = df_combo.sort_values("f1_mean", ascending=False)

# =====================================================================
# PERSISTÊNCIA
# =====================================================================
df_cv.to_csv("./resultados/metricas_cv.csv")
df_teste.to_csv("./resultados/metricas_teste.csv")
df_combo.to_csv("./resultados/metricas_completo.csv")

# =====================================================================
# EXIBIÇÃO
# =====================================================================
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
WIDTH = 90

# ---------- 1) CV ----------
print("\n" + "=" * WIDTH)
print("TREINO — Validação Cruzada 5-fold".center(WIDTH))
print("=" * WIDTH)

metricas_taxa = ["accuracy", "precision", "sensitivity", "specificity", "f1", "mh"]
resumo_cv = pd.DataFrame(index=df_cv.index)
for m in metricas_taxa:
    if f"{m}_mean" in df_cv.columns:
        resumo_cv[m] = df_cv.apply(
            lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}", axis=1
        )
print(resumo_cv.to_string())

# ---------- 2) TESTE ----------
print("\n" + "=" * WIDTH)
print("TESTE — modelo retreinado em X_train inteiro".center(WIDTH))
print("=" * WIDTH)

resumo_teste = pd.DataFrame(index=df_teste.index)
for col in df_teste.columns:
    if col.endswith("_f1"):
        prefixo = col.replace("_f1", "")
        resumo_teste[prefixo] = df_teste.apply(
            lambda r: f"f1={r[f'{prefixo}_f1']:.4f}  acc={r[f'{prefixo}_accuracy']:.4f}",
            axis=1,
        )
print(resumo_teste.to_string())

# ---------- 3) COMPARAÇÃO ----------
print("\n" + "=" * WIDTH)
print("COMPARAÇÃO — CV vs Teste".center(WIDTH))
print("=" * WIDTH)

comp = pd.DataFrame(index=df_combo.index)
comp["f1_CV"] = df_combo["f1_mean"].round(4)
for col in df_combo.columns:
    if col.endswith("_f1") and not col.startswith("delta_") and col != "f1_mean":
        nome = col.replace("_f1", "")
        comp[nome] = df_combo[col].round(4)
        comp[f"Δ_{nome}"] = df_combo[f"delta_{nome}_f1"].round(4)
print(comp.to_string())
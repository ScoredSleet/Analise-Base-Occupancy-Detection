import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from plots import plot_curva, plot_matrizes_confusao, plot_barras


# =====================================================================
# 1) Carregamento
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "dataset", "occupancy_detection", "dataset_csv")
if not os.path.isdir(CSV_DIR):
    CSV_DIR = os.path.join(
        BASE_DIR, "Analise-Base-Occupancy-Detection",
        "dataset", "occupancy_detection", "dataset_csv",
    )

train_df  = pd.read_csv(os.path.join(CSV_DIR, "datatraining.csv"))
test1_df  = pd.read_csv(os.path.join(CSV_DIR, "datatest.csv"))
test2_df  = pd.read_csv(os.path.join(CSV_DIR, "datatest2.csv"))

colunas_X = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

X_train = train_df[colunas_X].astype(float)
y_train = train_df["Occupancy"].astype(int)
X_test1 = test1_df[colunas_X].astype(float)
y_test1 = test1_df["Occupancy"].astype(int)
X_test2 = test2_df[colunas_X].astype(float)
y_test2 = test2_df["Occupancy"].astype(int)

# Z-score ajustado SÓ no treino (evita leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)
X_test2_scaled = scaler.transform(X_test2)


# =====================================================================
# 2) Métricas
# =====================================================================
def calcular_metricas(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Acurácia":      accuracy_score(y_true, y_pred),
        "Precisão":      precision_score(y_true, y_pred),
        "Sensibilidade": recall_score(y_true, y_pred),
        "Especificidade": tn / (tn + fp),
        "F1-score":      f1_score(y_true, y_pred),
        "cm":            cm,
    }


# =====================================================================
# 3) Experimentos (k × distância)
# =====================================================================
valores_k     = [1, 3, 5, 11, 21, 51, 91, 151]
metricas_dist = ["euclidean", "manhattan"]

resultados = []
for metric in metricas_dist:
    for k in valores_k:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train_scaled, y_train)

        m1 = calcular_metricas(y_test1, knn.predict(X_test1_scaled))
        m2 = calcular_metricas(y_test2, knn.predict(X_test2_scaled))

        resultados.append({
            "Metrica": metric, "k": k,
            "T1_Acc":   m1["Acurácia"],   "T1_Prec":  m1["Precisão"],
            "T1_Sens":  m1["Sensibilidade"], "T1_Espec": m1["Especificidade"],
            "T1_F1":    m1["F1-score"],   "T1_CM":    m1["cm"],
            "T2_Acc":   m2["Acurácia"],   "T2_Prec":  m2["Precisão"],
            "T2_Sens":  m2["Sensibilidade"], "T2_Espec": m2["Especificidade"],
            "T2_F1":    m2["F1-score"],   "T2_CM":    m2["cm"],
        })

res_df = pd.DataFrame(resultados)

print("=== TABELA COMPARATIVA KNN (datatest1) ===")
print(res_df[["Metrica", "k", "T1_Acc", "T1_Prec", "T1_Sens", "T1_Espec", "T1_F1"]]
      .to_string(index=False, float_format="{:.4f}".format))

print("\n=== TABELA COMPARATIVA KNN (datatest2) ===")
print(res_df[["Metrica", "k", "T2_Acc", "T2_Prec", "T2_Sens", "T2_Espec", "T2_F1"]]
      .to_string(index=False, float_format="{:.4f}".format))


# =====================================================================
# 4) Plots — via plots.py
# =====================================================================

# (1) Curva: 1 figura por distância (cada figura tem subplots T1 e T2)
plot_curva(
    df=res_df, x_col="k", group_col="Metrica",
    grupos=["euclidean", "manhattan"],
    nomes_grupos={"euclidean": "Distância Euclidiana",
                  "manhattan": "Distância Manhattan"},
    arquivo={
        "euclidean": "knn_curva_k_euclidiana.png",
        "manhattan": "knn_curva_k_manhattan.png",
    },
    titulo="KNN",
    marca_x=91,
    marca_label="k = 91 (≈ √N)",
)

# (2) Matrizes de confusão: 1 figura por distância (k=5 e k=91 × T1 e T2)
plot_matrizes_confusao(
    df=res_df, cm_col="CM",
    configs=[
        ({"Metrica": "euclidean", "k": 5},  "k=5 (Euclidiana)"),
        ({"Metrica": "euclidean", "k": 91}, "k=91 (Euclidiana)"),
    ],
    titulo="Matrizes de Confusão — KNN (Distância Euclidiana)",
    arquivo="knn_matrizes_confusao_euclidiana.png",
)
plot_matrizes_confusao(
    df=res_df, cm_col="CM",
    configs=[
        ({"Metrica": "manhattan", "k": 5},  "k=5 (Manhattan)"),
        ({"Metrica": "manhattan", "k": 91}, "k=91 (Manhattan)"),
    ],
    titulo="Matrizes de Confusão — KNN (Distância Manhattan)",
    arquivo="knn_matrizes_confusao_manhattan.png",
)

# (3) Barras: 5 métricas × 4 configs
plot_barras(
    df=res_df,
    configs=[
        ({"Metrica": "euclidean", "k": 5},  "k=5 (Euclidiana)"),
        ({"Metrica": "manhattan", "k": 5},  "k=5 (Manhattan)"),
        ({"Metrica": "euclidean", "k": 91}, "k=91 (Euclidiana)"),
        ({"Metrica": "manhattan", "k": 91}, "k=91 (Manhattan)"),
    ],
    titulo="Comparação das 5 Medidas de Avaliação no KNN",
    arquivo="knn_metricas_comparativo.png",
)

print("\nGráficos salvos com sucesso:")
for nome in ("knn_curva_k_euclidiana.png", "knn_curva_k_manhattan.png",
             "knn_matrizes_confusao_euclidiana.png",
             "knn_matrizes_confusao_manhattan.png",
             "knn_metricas_comparativo.png"):
    print(f"- {nome}")

plt.show()
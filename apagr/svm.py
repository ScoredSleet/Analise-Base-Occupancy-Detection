"""SVM no conjunto Occupancy Detection — refatorado com plots.py.

Estrutura igual ao knn.py:
- Carrega os 3 CSVs via carregar_base (formatar=True → inclui temporais)
- Para cada (escalonamento × kernel × C × gamma): fit no X_train, prediz T1/T2
- Gera 3 plots via plots.py (curva por escalonamento, top-4 barras, matrizes)
"""

import os
from itertools import product

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from preprocessamento import carregar_base
from plots import plot_curva, plot_matrizes_confusao, plot_barras


RAIZ = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(RAIZ, "dataset", "occupancy_detection", "dataset_csv")

# Mesma grade do svm.py original
ESCALONAMENTOS = ["Z-Score", "Min-Max", "None"]
PARAMETROS_SVM = [
    ("linear", 0.1, "scale"),
    ("linear", 1.0, "scale"),
    ("rbf", 1.0, "scale"),
    ("rbf", 1.0, "auto"),
    ("rbf", 10.0, "scale"),
]


def carregar(nome_arquivo):
    X, y = carregar_base(
        os.path.join(CSV_DIR, nome_arquivo),
        return_X_y=True, formatar=True,
    )
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"Valores ausentes em {nome_arquivo}.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Classes inesperadas em {nome_arquivo}.")
    return X, y


def criar_modelo(escalonamento, kernel, C, gamma):
    escala = {
        "Z-Score": StandardScaler(),
        "Min-Max": MinMaxScaler(),
        "None":    "passthrough",
    }[escalonamento]
    return Pipeline([
        ("escala", escala),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma)),
    ])


def calcular_metricas(y_real, y_previsto):
    cm = confusion_matrix(y_real, y_previsto, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Acc":   accuracy_score(y_real, y_previsto),
        "Prec":  precision_score(y_real, y_previsto, zero_division=0),
        "Sens":  recall_score(y_real, y_previsto, zero_division=0),
        "Espec": tn / (tn + fp) if tn + fp else 0.0,
        "F1":    f1_score(y_real, y_previsto, zero_division=0),
        "cm":    cm,
    }


def main():
    X_train, y_train = carregar("datatraining.csv")
    X_test1, y_test1 = carregar("datatest.csv")
    X_test2, y_test2 = carregar("datatest2.csv")

    for nome, X in [("datatest", X_test1), ("datatest2", X_test2)]:
        if list(X.columns) != list(X_train.columns):
            raise ValueError(f"Colunas diferentes em {nome}.")

    print(f"SVM | treino: {len(y_train)} | datatest: {len(y_test1)} | datatest2: {len(y_test2)}")

    # ------------------ EXPERIMENTOS ------------------
    resultados = []
    for escalonamento, (kernel, C, gamma) in product(ESCALONAMENTOS, PARAMETROS_SVM):
        modelo = criar_modelo(escalonamento, kernel, C, gamma)
        modelo.fit(X_train, y_train)

        m1 = calcular_metricas(y_test1, modelo.predict(X_test1))
        m2 = calcular_metricas(y_test2, modelo.predict(X_test2))

        if kernel == "linear":
            param_label = f"linear\nC={C:g}"
            gamma_label = "—"
        else:
            param_label = f"rbf\nC={C:g}\nγ={gamma}"
            gamma_label = gamma

        resultados.append({
            "Escalonamento": escalonamento,
            "Kernel":        kernel,
            "C":             C,
            "Gamma":         gamma_label,
            "Parametro":     param_label,
            "T1_Acc": m1["Acc"], "T1_Prec": m1["Prec"], "T1_Sens": m1["Sens"],
            "T1_Espec": m1["Espec"], "T1_F1": m1["F1"], "T1_CM": m1["cm"],
            "T2_Acc": m2["Acc"], "T2_Prec": m2["Prec"], "T2_Sens": m2["Sens"],
            "T2_Espec": m2["Espec"], "T2_F1": m2["F1"], "T2_CM": m2["cm"],
        })

    res_df = pd.DataFrame(resultados)

    # ------------------ TABELAS ------------------
    print("\n=== TABELA COMPARATIVA SVM (datatest1) ===")
    print(res_df[["Escalonamento", "Kernel", "C", "Gamma",
                  "T1_Acc", "T1_Prec", "T1_Sens", "T1_Espec", "T1_F1"]]
          .to_string(index=False, float_format="{:.4f}".format))

    print("\n=== TABELA COMPARATIVA SVM (datatest2) ===")
    print(res_df[["Escalonamento", "Kernel", "C", "Gamma",
                  "T2_Acc", "T2_Prec", "T2_Sens", "T2_Espec", "T2_F1"]]
          .to_string(index=False, float_format="{:.4f}".format))

    os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
    res_df.drop(columns=["Parametro"]).to_csv(
        os.path.join(RAIZ, "resultados", "svm_completo.csv"), index=False,
    )

    # ------------------ PLOTS ------------------

    # (1) Curva — 1 figura por escalonamento, 5 configs no X
    plot_curva(
        df=res_df, x_col="Parametro", group_col="Escalonamento",
        grupos=["Z-Score", "Min-Max", "None"],
        nomes_grupos={"Z-Score": "Z-Score",
                      "Min-Max": "Min-Max",
                      "None":    "Sem escalonamento"},
        arquivo={
            "Z-Score": "svm_curva_zscore.png",
            "Min-Max": "svm_curva_minmax.png",
            "None":    "svm_curva_none.png",
        },
        titulo="SVM",
        xlabel="Configuração (kernel / C / γ)",
        y_lim=(0.70, 1.02),
    )

    # (2) Barras — top 4 por F1 no datatest2
    top4 = res_df.nlargest(4, "T2_F1")
    configs_top4 = [
        ({"Escalonamento": r["Escalonamento"], "Parametro": r["Parametro"]},
         f"{r['Escalonamento']} · {r['Kernel']} C={r['C']}"
         + (f" γ={r['Gamma']}" if r["Kernel"] == "rbf" else ""))
        for _, r in top4.iterrows()
    ]
    plot_barras(
        df=res_df, configs=configs_top4,
        titulo="SVM — Top 4 configurações (por F1 no datatest2)",
        arquivo="svm_metricas_comparativo.png",
        y_lim=(0.70, 1.05),
    )

    # (3) Matrizes — melhor de cada escalonamento
    configs_best = []
    for escala in ["Z-Score", "Min-Max", "None"]:
        sub = res_df[res_df["Escalonamento"] == escala]
        r = sub.loc[sub["T2_F1"].idxmax()]
        label = f"{escala} · {r['Kernel']} C={r['C']}"
        if r["Kernel"] == "rbf":
            label += f" γ={r['Gamma']}"
        configs_best.append(
            ({"Escalonamento": escala, "Parametro": r["Parametro"]}, label)
        )

    plot_matrizes_confusao(
        df=res_df, configs=configs_best, cm_col="CM",
        titulo="SVM — Matrizes de Confusão (melhor de cada escalonamento)",
        arquivo="svm_matrizes_confusao.png",
        cmap="Blues",
    )

    print("\nGráficos salvos com sucesso.")
    print("\nTop 5 configs (por T2_F1):")
    print(res_df.nlargest(5, "T2_F1")
          [["Escalonamento", "Kernel", "C", "Gamma", "T1_F1", "T2_F1"]]
          .to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
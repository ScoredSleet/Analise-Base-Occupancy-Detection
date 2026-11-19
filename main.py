"""

Base:https://archive.ics.uci.edu/dataset/357/occupancy+detection
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from metricas import treino, validacao

import pandas as pd
import os

PATH_TREINO = "./dataset/occupancy_detection/dataset_csv/datatraining.csv"
PATH_TEST = "./dataset/occupancy_detection/dataset_csv/datatest.csv"
PATH_TEST2 = "./dataset/occupancy_detection/dataset_csv/datatest2.csv"
PATHS_TESTE = [PATH_TEST, PATH_TEST2]

RANDOM_STATE = 42

models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=RANDOM_STATE),
    "XGB": XGBClassifier(random_state=RANDOM_STATE),
    "NB": GaussianNB(),
}

scalers = {
    "Z-Score": StandardScaler(),
    "Min-Max": MinMaxScaler(),
    "None": "passthrough",
}


def salvar_resultados(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    df_combo: pd.DataFrame,
    pasta_saida: str = "./resultados",
    nome_arq: str = "teste",
) -> None:
    """Salva os DataFrames de métricas em arquivos CSV."""
    os.makedirs(pasta_saida, exist_ok=True)
    df_treino.to_csv(os.path.join(pasta_saida, f"metricas_{nome_arq}_treino.csv"))
    df_teste.to_csv(os.path.join(pasta_saida, f"metricas_{nome_arq}_teste.csv"))
    df_combo.to_csv(os.path.join(pasta_saida, f"metricas_{nome_arq}_completo.csv"))


def processar_resultados(
    resultados_treino: dict,
    resultados_teste: dict,
    pasta_saida: str = "./resultados",
    nome_arq: str = "teste",
    salvar: bool = True,
    exibir: bool = True,
    width: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constrói os DataFrames, salva os resultados e exibe o relatório no terminal."""
    # =====================================================================
    # 1) CONSTRUÇÃO DOS DATAFRAMES
    # =====================================================================
    df_treino = pd.DataFrame(resultados_treino).T
    df_treino.index.name = "Modelos"
    df_treino = df_treino.sort_values("f1", ascending=False)

    df_teste = pd.DataFrame(resultados_teste).T
    df_teste.index.name = "modelo_scaler"
    df_teste = df_teste.reindex(df_treino.index)

    df_combo = df_treino.join(df_teste)
    df_combo = df_combo.sort_values("f1", ascending=False)

    # =====================================================================
    # 2) PERSISTÊNCIA
    # =====================================================================
    if salvar:
        salvar_resultados(
            df_treino, df_teste, df_combo, pasta_saida=pasta_saida, nome_arq=nome_arq
        )

    # =====================================================================
    # 3) EXIBIÇÃO
    # =====================================================================
    if exibir:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 50)

        # ---------- TREINO (100% DATATRAINING) ----------
        print("\n" + "=" * width)
        print("TREINO — Conjunto Completo (datatraining)".center(width))
        print("=" * width)

        metricas_taxa = ["accuracy", "precision", "sensitivity", "specificity", "f1", "mh"]
        colunas_presentes = [m for m in metricas_taxa if m in df_treino.columns]
        print(df_treino[colunas_presentes].round(4).to_string())

        # ---------- TESTE ----------
        print("\n" + "=" * width)
        print("TESTE — Datasets Inéditos (datatest e datatest2)".center(width))
        print("=" * width)

        resumo_teste = pd.DataFrame(index=df_teste.index)
        colunas_ordenar = []
        for col in df_teste.columns:
            if col.endswith("_f1"):
                prefixo = col.removesuffix("_f1")
                resumo_teste[prefixo] = df_teste.apply(
                    lambda r: f"f1={r[f'{prefixo}_f1']:.4f}  acc={r[f'{prefixo}_accuracy']:.4f}",
                    axis=1,
                )
                colunas_ordenar.append(col)

        if colunas_ordenar:
            ordem_teste = (
                df_teste[colunas_ordenar].mean(axis=1).sort_values(ascending=False).index
            )
            resumo_teste = resumo_teste.reindex(ordem_teste)
        print(resumo_teste.to_string())

        # ---------- COMPARAÇÃO F1-SCORE ----------
        print("\n" + "=" * width)
        print("COMPARAÇÃO — Treino vs Teste | F1-Score".center(width))
        print("=" * width)

        comp_f1 = pd.DataFrame(index=df_combo.index)
        comp_f1["f1_treino"] = df_combo["f1"].round(4)

        colunas_f1_teste = []
        for col in df_combo.columns:
            if col.endswith("_f1") and col != "f1":
                nome = col.removesuffix("_f1")
                comp_f1[nome] = df_combo[col].round(4)
                colunas_f1_teste.append(nome)

        if colunas_f1_teste:
            ordem_f1 = comp_f1[colunas_f1_teste].mean(axis=1).sort_values(ascending=False).index
            comp_f1 = comp_f1.reindex(ordem_f1)

        print(comp_f1.to_string())

        # ---------- COMPARAÇÃO ACURÁCIA ----------
        print("\n" + "=" * width)
        print("COMPARAÇÃO — Treino vs Teste | Acurácia".center(width))
        print("=" * width)

        comp_acc = pd.DataFrame(index=df_combo.index)
        comp_acc["accuracy_treino"] = df_combo["accuracy"].round(4)

        colunas_acc_teste = []
        for col in df_combo.columns:
            if col.endswith("_accuracy") and col != "accuracy":
                nome = col.removesuffix("_accuracy")
                comp_acc[nome] = df_combo[col].round(4)
                colunas_acc_teste.append(nome)

        if colunas_acc_teste:
            ordem_acc = comp_acc[colunas_acc_teste].mean(axis=1).sort_values(ascending=False).index
            comp_acc = comp_acc.reindex(ordem_acc)

        print(comp_acc.to_string())

    return df_treino, df_teste, df_combo


def main():
    resultados_treino, modelos_treinados = treino(
        models=models,
        scalers=scalers,
        PATH_treino=PATH_TREINO,
        formatar=True,
        verbose=True,
    )

    resultados_teste = validacao(
        modelos_treinados=modelos_treinados,
        PATHs_teste=PATHS_TESTE,
        formatar=True,
        verbose=True,
    )

    df_treino, df_teste, df_combo = processar_resultados(resultados_treino, resultados_teste)


if __name__ == "__main__":
    main()
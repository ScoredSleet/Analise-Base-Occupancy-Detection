"""XGBoost no conjunto Occupancy Detection.

Execute na raiz do projeto: python modelo_xgboost.py
Dependencias: preprocessamento.py, pandas, scikit-learn e xgboost.

1) Reutiliza o pre-processamento existente, sem escalonamento adicional.
2) Divide datatraining em treino (80%) e validacao (20%), na ordem temporal.
3) Testa hiperparametros e seleciona pelo F1 da validacao.
4) Reajusta a melhor configuracao em datatraining completo.
5) Avalia datatest e datatest2 sem usar esses conjuntos para escolher parametros.
"""

from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from preprocessamento import carregar_base


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dataset" / "occupancy_detection" / "dataset_csv"
RESULTADOS = RAIZ / "resultados"

# Oito configuracoes: 2 valores para cada um dos tres hiperparametros.
N_ESTIMATORS = (50, 100)
MAX_DEPTH = (2, 4)
LEARNING_RATE = (0.1, 0.2)


def carregar(nome_arquivo):
    """Usa o pre-processamento comum; mantem os atributos sem escala adicional."""
    X, y = carregar_base(DADOS / nome_arquivo, return_X_y=True, formatar=True)
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"Valores ausentes em {nome_arquivo}.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Classes inesperadas em {nome_arquivo}.")
    return X, y


def criar_modelo(n_estimators, max_depth, learning_rate):
    """Arvores nao exigem StandardScaler/MinMaxScaler para esta comparacao."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def calcular_metricas(y_real, y_previsto):
    tn, fp, fn, tp = confusion_matrix(y_real, y_previsto, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y_real, y_previsto, zero_division=0),
        "acuracia": accuracy_score(y_real, y_previsto),
        "precisao": precision_score(y_real, y_previsto, zero_division=0),
        "sensibilidade": recall_score(y_real, y_previsto, zero_division=0),
        "especificidade": tn / (tn + fp) if tn + fp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    X, y = carregar("datatraining.csv")
    corte = int(len(y) * 0.80)
    if corte <= 0 or corte >= len(y):
        raise ValueError("Quantidade de registros insuficiente para a divisao 80/20.")

    # preprocessamento.py ja ordenou os dados por data; nao embaralhamos.
    X_treino, X_val = X.iloc[:corte], X.iloc[corte:]
    y_treino, y_val = y.iloc[:corte], y.iloc[corte:]
    print(f"XGBoost | treino interno: {len(y_treino)} | validacao: {len(y_val)}")
    print("Escalonamento adicional: nenhum (modelo baseado em arvores).")

    experimentos = []
    for n, profundidade, taxa in product(N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE):
        modelo = criar_modelo(n, profundidade, taxa)
        modelo.fit(X_treino, y_treino)
        metricas = calcular_metricas(y_val, modelo.predict(X_val))
        experimento = {
            "n_estimators": n, "max_depth": profundidade,
            "learning_rate": taxa, **metricas,
        }
        experimentos.append(experimento)
        print(f"  arvores={n:<3} | profundidade={profundidade} | "
              f"taxa={taxa:.1f} | F1(val)={metricas['f1']:.4f}")

    # Somente resultados da validacao entram na selecao.
    # Em empate de F1 e sensibilidade, prevalece a primeira configuracao
    # (menos arvores, profundidade menor e taxa menor).
    tabela = pd.DataFrame(experimentos).sort_values(
        ["f1", "sensibilidade"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    caminho_validacao = RESULTADOS / "xgboost_simples_validacao.csv"
    tabela.to_csv(caminho_validacao, index=False)

    melhor = tabela.iloc[0]
    print("\nMelhor configuracao na validacao:")
    print(f"  n_estimators={int(melhor['n_estimators'])}, "
          f"max_depth={int(melhor['max_depth'])}, "
          f"learning_rate={melhor['learning_rate']}, F1={melhor['f1']:.4f}")

    # Somente agora treinamos o modelo definitivo e abrimos os conjuntos de teste.
    modelo_final = criar_modelo(int(melhor["n_estimators"]),
                               int(melhor["max_depth"]),
                               float(melhor["learning_rate"]))
    modelo_final.fit(X, y)

    testes = []
    for arquivo in ("datatest.csv", "datatest2.csv"):
        X_teste, y_teste = carregar(arquivo)
        if list(X_teste.columns) != list(X.columns):
            raise ValueError(f"Colunas diferentes em {arquivo}.")
        metricas = calcular_metricas(y_teste, modelo_final.predict(X_teste))
        testes.append({"conjunto": arquivo, **metricas})
        print(f"\n{arquivo}: F1={metricas['f1']:.4f} | "
              f"Acuracia={metricas['acuracia']:.4f} | "
              f"Precisao={metricas['precisao']:.4f} | "
              f"Sensibilidade={metricas['sensibilidade']:.4f} | "
              f"Especificidade={metricas['especificidade']:.4f}")
        print("Matriz de confusao [[TN, FP], [FN, TP]]:")
        print(f"  [[{metricas['tn']}, {metricas['fp']}], "
              f"[{metricas['fn']}, {metricas['tp']}]]")

    caminho_testes = RESULTADOS / "xgboost_simples_testes.csv"
    pd.DataFrame(testes).to_csv(caminho_testes, index=False)
    print(f"\nResultados salvos em:\n  {caminho_validacao}\n  {caminho_testes}")


if __name__ == "__main__":
    main()

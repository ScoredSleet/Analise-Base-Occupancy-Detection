"""SVM no conjunto Occupancy Detection.

Execute na raiz do projeto: python svm.py
Dependencias: preprocessamento.py, pandas e scikit-learn.

1) Reutiliza o pre-processamento existente.
2) Divide datatraining em treino (80%) e validacao (20%), na ordem temporal.
3) Testa escalonamentos e hiperparametros, selecionando pelo F1 da validacao.
4) Reajusta a melhor configuracao em datatraining completo.
5) Avalia datatest e datatest2 sem usar esses conjuntos para escolher parametros.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from preprocessamento import carregar_base


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dataset" / "occupancy_detection" / "dataset_csv"
RESULTADOS = RAIZ / "resultados"

# Comparacao pequena e predefinida. Outras combinacoes podem ter desempenho melhor.
# Em empate exato de F1 e sensibilidade, mantemos a primeira configuracao:
# Z-Score, depois Min-Max e finalmente sem escalonamento.
ESCALONAMENTOS = ("Z-Score", "Min-Max", "None")
PARAMETROS_SVM = (
    ("linear", 0.1, "scale"),
    ("linear", 1.0, "scale"),
    ("rbf", 1.0, "scale"),
    ("rbf", 1.0, "auto"),
    ("rbf", 10.0, "scale"),
)


def carregar(nome_arquivo):
    """Usa o pre-processamento comum; formatar=True cria atributos temporais."""
    X, y = carregar_base(DADOS / nome_arquivo, return_X_y=True, formatar=True)
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"Valores ausentes em {nome_arquivo}.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Classes inesperadas em {nome_arquivo}.")
    return X, y


def criar_modelo(escalonamento, kernel, C, gamma):
    """O escalonador e ajustado pelo Pipeline somente nos dados do fit."""
    if escalonamento == "Z-Score":
        escala = StandardScaler()
    elif escalonamento == "Min-Max":
        escala = MinMaxScaler()
    else:
        escala = "passthrough"

    return Pipeline([
        ("escala", escala),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma)),
    ])


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
    print(f"SVM | treino interno: {len(y_treino)} | validacao: {len(y_val)}")

    experimentos = []
    for escala in ESCALONAMENTOS:
        for kernel, C, gamma in PARAMETROS_SVM:
            modelo = criar_modelo(escala, kernel, C, gamma)
            modelo.fit(X_treino, y_treino)
            metricas = calcular_metricas(y_val, modelo.predict(X_val))
            experimento = {
                "escalonamento": escala,
                "kernel": kernel,
                "C": C,
                "gamma": gamma if kernel != "linear" else "nao_se_aplica",
                **metricas,
            }
            experimentos.append(experimento)
            print(f"  {escala:7} | kernel={kernel:6} | C={C:<4g} | "
                  f"gamma={experimento['gamma']:13} | F1(val)={metricas['f1']:.4f}")

    # Somente resultados da validacao entram na selecao.
    tabela = pd.DataFrame(experimentos).sort_values(
        ["f1", "sensibilidade"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    caminho_validacao = RESULTADOS / "svm_simples_validacao.csv"
    tabela.to_csv(caminho_validacao, index=False)

    melhor = tabela.iloc[0]
    print("\nMelhor configuracao na validacao:")
    print(f"  Escalonamento={melhor['escalonamento']}, kernel={melhor['kernel']}, "
          f"C={melhor['C']}, gamma={melhor['gamma']}, F1={melhor['f1']:.4f}")

    # Somente agora treinamos o modelo definitivo e abrimos os conjuntos de teste.
    gamma = melhor["gamma"] if melhor["gamma"] != "nao_se_aplica" else "scale"
    modelo_final = criar_modelo(melhor["escalonamento"], melhor["kernel"],
                               float(melhor["C"]), gamma)
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

    caminho_testes = RESULTADOS / "svm_simples_testes.csv"
    pd.DataFrame(testes).to_csv(caminho_testes, index=False)
    print(f"\nResultados salvos em:\n  {caminho_validacao}\n  {caminho_testes}")


if __name__ == "__main__":
    main()

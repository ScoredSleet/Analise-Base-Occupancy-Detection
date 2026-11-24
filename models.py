"""
models.py — pipeline unificado para todos os modelos do projeto.

Uso:
    from models import rodar_todos
    resultados = rodar_todos()          # roda KNN, SVM, NB, XGB
    resultados = rodar_todos(["KNN"])   # roda só KNN

Cada modelo gera:
    - um DataFrame com uma linha por (config × scaler)
    - um CSV em ./resultados/<modelo>_completo.csv
    - 3 PNGs em ./plots/ (curva, barras, matrizes)
"""

import os
from itertools import product

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier

from preprocessamento import carregar_base
from plots import plot_curva, plot_matrizes_confusao, plot_barras


# =====================================================================
# CAMINHOS
# =====================================================================
RAIZ            = os.path.dirname(os.path.abspath(__file__))
CSV_DIR         = os.path.join(RAIZ, "dataset", "occupancy_detection", "dataset_csv")
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")
PASTA_PLOTS      = os.path.join(RAIZ, "plots")


# =====================================================================
# CONFIGURAÇÃO DOS MODELOS
# =====================================================================

_SVM_PARAMS = [
    ("linear", 0.1, "scale"),
    ("linear", 1.0, "scale"),
    ("rbf", 1.0, "scale"),
    ("rbf", 1.0, "auto"),
    ("rbf", 10.0, "scale"),
]

MODELOS = {
    "KNN": {
        "classe": KNeighborsClassifier,
        "configs": [
            {"n_neighbors": k, "metric": m}
            for k in [1, 3, 5, 11, 21, 51, 91, 151]
            for m in ["euclidean", "manhattan"]
        ],
        "scalers": {"Z-Score": StandardScaler,
                    "Min-Max": MinMaxScaler,
                    "None": None},
        "formatar": False,
        "x_col": "label",
        "x_label": "Número de vizinhos (k) e Métrica",  
        "group_col": None,                                 
        "nomes_grupos": {},
        "marca_x": None,                                   
        "marca_label": None,
        "top_n_barras": 4,
        "params_fixos": {},
    },
    "SVM": {
        "classe": SVC,
        "configs": [
            {"kernel": k, "C": c, "gamma": g}
            for (k, c, g) in _SVM_PARAMS
        ],
        "scalers": {"Z-Score": StandardScaler,
                    "Min-Max": MinMaxScaler,
                    "None": None},
        "formatar": True,
        "x_col": "label",
        "x_label": "Configuração (kernel / C / γ)",
        "group_col": "escalonamento",
        "nomes_grupos": {"Z-Score": "Z-Score",
                         "Min-Max": "Min-Max",
                         "None": "Sem escalonamento"},
        "marca_x": None,
        "marca_label": None,
        "top_n_barras": 4,
        "params_fixos": {"random_state": 42},
    },
    "NB": {
        "classe": GaussianNB,
        "configs": [{"var_smoothing": v}
                    for v in [1e-9, 1e-3, 1e-2, 0.1, 0.5, 1.0]],
        "scalers": {"Z-Score": StandardScaler,
                    "Min-Max": MinMaxScaler,
                    "None": None
                },
        "formatar": False,
        "x_col": "var_smoothing",
        "x_label": "Parâmetro var_smoothing",
        "group_col": None,
        "nomes_grupos": {},
        "marca_x": None,
        "marca_label": None,
        "top_n_barras": 3,
        "params_fixos": {},
    },
    "XGB": {
        "classe": XGBClassifier,
        "configs": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n, d, lr in product([50, 100], [2, 4], [0.1, 0.2])
        ],
        "scalers": {"None": None},
        "formatar": True,
        "x_col": "label",
        "x_label": "Configuração (árvores / prof. / lr)",
        "group_col": None,
        "nomes_grupos": {},
        "marca_x": None,
        "marca_label": None,
        "top_n_barras": 4,
        "params_fixos": {"random_state": 42, "n_jobs": 1,
                         "verbosity": 0, "tree_method": "hist"},
    },
}


# =====================================================================
# CARREGAMENTO E MÉTRICAS
# =====================================================================

def carregar(nome_arquivo, formatar):
    X, y = carregar_base(
        os.path.join(CSV_DIR, nome_arquivo),
        return_X_y=True, formatar=formatar,
    )
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"Valores ausentes em {nome_arquivo}.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Classes inesperadas em {nome_arquivo}.")
    return X, y


def calcular_metricas(y_real, y_previsto):
    """Acc/Prec/Sens/Espec/F1 + matriz de confusão (chaves fixas)."""
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


# =====================================================================
# CONSTRUÇÃO DO PIPELINE E LABELS
# =====================================================================

def criar_pipeline(classe_modelo, scaler_cls, params_fixos=None, **params):
    """Pipeline(scaler, modelo). scaler_cls=None → passthrough."""
    escala = "passthrough" if scaler_cls is None else scaler_cls()
    params_completos = {**(params_fixos or {}), **params}
    return Pipeline([
        ("escala", escala),
        ("modelo", classe_modelo(**params_completos)),
    ])


def criar_label(nome_modelo, config):
    """Rótulo legível da config (multi-linha para o eixo X do plot de curva)."""
    if nome_modelo == "KNN":
        m = "Eucl" if config["metric"] == "euclidean" else "Manh"
        return f"k={config['n_neighbors']}\n({m})"
    if nome_modelo == "SVM":
        if config["kernel"] == "linear":
            return f"linear\nC={config['C']:g}"
        return f"rbf\nC={config['C']:g}\nγ={config['gamma']}"
    if nome_modelo == "NB":
        vs = config["var_smoothing"]
        return f"{vs:.0e}" if vs < 0.01 else f"{vs:g}"
    if nome_modelo == "XGB":
        return (f"{config['n_estimators']} árv.\n"
                f"d={config['max_depth']}\n"
                f"lr={config['learning_rate']:g}")
    return str(config)


# =====================================================================
# RODAR UM MODELO (config × scaler × test)
# =====================================================================

def rodar_modelo(nome_modelo, cfg, dados):
    """Itera configs × scalers e retorna DataFrame com T1_*/T2_*."""
    (X_train, y_train) = dados["train"]
    (X_test1, y_test1) = dados["test1"]
    (X_test2, y_test2) = dados["test2"]

    resultados = []
    for scaler_nome, scaler_cls in cfg["scalers"].items():
        for idx, config in enumerate(cfg["configs"]):
            pipe = criar_pipeline(
                cfg["classe"], scaler_cls,
                params_fixos=cfg.get("params_fixos"),
                **config,
            )
            pipe.fit(X_train, y_train)

            m1 = calcular_metricas(y_test1, pipe.predict(X_test1))
            m2 = calcular_metricas(y_test2, pipe.predict(X_test2))

            label = criar_label(nome_modelo, config)
            row_id = f"{nome_modelo}|{scaler_nome}|{idx}"

            resultados.append({
                **config,
                "escalonamento": scaler_nome,
                "label":         label,
                "id":            row_id,
                "T1_Acc":  m1["Acc"],  "T1_Prec": m1["Prec"],
                "T1_Sens": m1["Sens"], "T1_Espec": m1["Espec"],
                "T1_F1":   m1["F1"],   "T1_CM":    m1["cm"],
                "T2_Acc":  m2["Acc"],  "T2_Prec": m2["Prec"],
                "T2_Sens": m2["Sens"], "T2_Espec": m2["Espec"],
                "T2_F1":   m2["F1"],   "T2_CM":    m2["cm"],
            })

    return pd.DataFrame(resultados)


# =====================================================================
# PLOTS POR MODELO  (salvando em ./plots/)
# =====================================================================

def gerar_plots(nome_modelo, res_df, cfg):
    prefixo = nome_modelo.lower()

    # (1) Curva ---------------------------------------------------
    if cfg["group_col"]:
        grupos = (["euclidean", "manhattan"] if cfg["group_col"] == "metric"
                  else list(cfg["scalers"].keys()))
        arquivo_curva = {
            g: os.path.join(PASTA_PLOTS, f"{prefixo}_curva_{g}.png")
            for g in grupos
        }
    else:
        grupos = [None]
        arquivo_curva = os.path.join(PASTA_PLOTS, f"{prefixo}_curva.png")

    plot_curva(
        df=res_df,
        x_col=cfg["x_col"],
        group_col=cfg["group_col"],
        grupos=grupos,
        nomes_grupos=cfg["nomes_grupos"],
        arquivo=arquivo_curva,
        titulo=nome_modelo,
        xlabel=cfg["x_label"],
        marca_x=cfg["marca_x"],
        marca_label=cfg["marca_label"],
    )

    # (2) Barras — top-N por T2_F1 -------------------------------
    top = res_df.nlargest(cfg["top_n_barras"], "T2_F1")
    configs_bar = [
        ({"id": r["id"]},
         f"{r['label'].replace(chr(10), ' ')} | {r['escalonamento']}"
         if len(cfg["scalers"]) > 1 else r["label"].replace(chr(10), " "))
        for _, r in top.iterrows()
    ]
    plot_barras(
        df=res_df,
        configs=configs_bar,
        titulo=f"{nome_modelo} — Top {cfg['top_n_barras']} (por F1 no datatest2)",
        arquivo=os.path.join(PASTA_PLOTS, f"{prefixo}_metricas_comparativo.png"),
    )

    # (3) Matrizes — melhor de cada escalonamento -----------------
    datasets_cm = [
        ("T1", "T1_F1", "T1_CM", "datatest"),
        ("T2", "T2_F1", "T2_CM", "datatest2"),
    ]

    for tag, f1_col, cm_col, nome_ds in datasets_cm:
        r = res_df.loc[res_df[f1_col].idxmax()]
        lbl = f"{r['label'].replace(chr(10), ' ')} | {r['escalonamento']}"
        configs_cm = [({"id": r["id"]}, lbl)]

        plot_matrizes_confusao(
            df=res_df,
            configs=configs_cm,
            cm_col=cm_col,
            titulo=f"{nome_modelo} — Melhor Matriz ({nome_ds})",
            arquivo=os.path.join(PASTA_PLOTS, f"{prefixo}_matriz_{nome_ds}.png"),
        )


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

def rodar_todos(modelos=None, salvar=True, plots=True):
    """Roda todos os modelos de MODELOS (ou só os listados)."""
    modelos = modelos or list(MODELOS.keys())
    todos = {}

    cache = {}
    def get_data(formatar):
        if formatar not in cache:
            cache[formatar] = {
                "train": carregar("datatraining.csv", formatar),
                "test1": carregar("datatest.csv",   formatar),
                "test2": carregar("datatest2.csv",  formatar),
            }
        return cache[formatar]

    for nome in modelos:
        cfg = MODELOS[nome]
        dados = get_data(cfg["formatar"])
        X_train, _ = dados["train"]
        X_test1, _ = dados["test1"]
        X_test2, _ = dados["test2"]

        for ds_nome, X in [("datatest", X_test1), ("datatest2", X_test2)]:
            if list(X.columns) != list(X_train.columns):
                raise ValueError(f"{nome}: colunas diferentes em {ds_nome}.")

        n_exp = len(cfg["configs"]) * len(cfg["scalers"])
        print(f"\n{'='*72}")
        print(f"Modelo: {nome}")
        print(f"{'='*72}")
        print(f"  features:   {len(X_train.columns)} (formatar={cfg['formatar']})")
        print(f"  configs:    {len(cfg['configs'])} × scalers: {len(cfg['scalers'])} = {n_exp} experimentos")

        res_df = rodar_modelo(nome, cfg, dados)
        todos[nome] = res_df

        if salvar:
            os.makedirs(PASTA_RESULTADOS, exist_ok=True)
            res_df.drop(columns=["T1_CM", "T2_CM", "id"]).to_csv(
                os.path.join(PASTA_RESULTADOS, f"{nome.lower()}_completo.csv"),
                index=False,
            )

        print("\n  Top 5 (por T2_F1):")
        print(res_df.nlargest(5, "T2_F1")[
            ["label", "escalonamento", "T1_F1", "T2_F1"]
        ].to_string(index=False, float_format="{:.4f}".format))

        if plots:
            gerar_plots(nome, res_df, cfg)

    return todos


if __name__ == "__main__":
    rodar_todos()
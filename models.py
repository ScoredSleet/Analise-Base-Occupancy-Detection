"""
models.py — pipeline unificado para todos os modelos do projeto.

Uso:
    from models import rodar_todos
    resultados = rodar_todos()          # roda KNN, SVM, NB, XGB
    resultados = rodar_todos(["KNN"])   # roda só KNN

Fluxo sem viés de seleção:
    1) Divide datatraining.csv em 80% treino interno e 20% validação (temporal).
    2) Avalia todas as configs × scalers na validação (Val_*).
    3) Seleciona e ordena os melhores modelos exclusivamente por Val_F1.
    4) Reajusta cada modelo em 100% de datatraining.csv e avalia em datatest (T1) e datatest2 (T2).

Cada modelo gera:
    - um DataFrame ordenado por Val_F1
    - CSVs em ./resultados/<modelo>_validacao.csv e ./resultados/<modelo>_completo.csv
    - PNGs em ./plots/ (curva, barras, matrizes)
"""

import os
from itertools import product

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from metricas import compute_metrics
from preprocessamento import carregar_base
from plots import plot_curva, plot_matrizes_confusao, plot_barras


# =====================================================================
# CAMINHOS
# =====================================================================
RAIZ             = os.path.dirname(os.path.abspath(__file__))
CSV_DIR          = os.path.join(RAIZ, "dataset", "occupancy_detection", "dataset_csv")
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
        "scalers": {
                    "Z-Score": StandardScaler,
                    "Min-Max": MinMaxScaler,
                    "None": None
                },
        "formatar": True,
        "x_col": "n_neighbors",
        "x_label": "Número de vizinhos (k)",
        "group_col": "metric",
        "nomes_grupos": {"euclidean": "Distância Euclidiana",
                         "manhattan": "Distância Manhattan"},
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
                    "None": None},
        "formatar": True,
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
# RODAR UM MODELO (80/20 Validação -> Reajuste 100% -> Testes)
# =====================================================================

def rodar_modelo(nome_modelo, cfg, dados):
    """Divide treino em 80/20 temporal para validar sem viés e reajusta em 100%."""
    (X_train, y_train) = dados["train"]
    (X_test1, y_test1) = dados["test1"]
    (X_test2, y_test2) = dados["test2"]

    corte = int(len(y_train) * 0.80)
    if corte <= 0 or corte >= len(y_train):
        raise ValueError("Quantidade de registros insuficiente para a divisão 80/20.")

    # Divisão temporal (sem embaralhar)
    X_tr_int, X_val = X_train.iloc[:corte], X_train.iloc[corte:]
    y_tr_int, y_val = y_train.iloc[:corte], y_train.iloc[corte:]

    resultados = []
    for scaler_nome, scaler_cls in cfg["scalers"].items():
        for idx, config in enumerate(cfg["configs"]):
            # 1) Treino interno (80%) e avaliação na validação (20%)
            pipe_val = criar_pipeline(
                cfg["classe"], scaler_cls,
                params_fixos=cfg.get("params_fixos"),
                **config,
            )
            pipe_val.fit(X_tr_int, y_tr_int)
            m_val = compute_metrics(y_val, pipe_val.predict(X_val))

            # 2) Reajuste em 100% do datatraining para avaliação final nos testes
            pipe_final = criar_pipeline(
                cfg["classe"], scaler_cls,
                params_fixos=cfg.get("params_fixos"),
                **config,
            )
            pipe_final.fit(X_train, y_train)
            m1 = compute_metrics(y_test1, pipe_final.predict(X_test1))
            m2 = compute_metrics(y_test2, pipe_final.predict(X_test2))

            label = criar_label(nome_modelo, config)
            row_id = f"{nome_modelo}|{scaler_nome}|{idx}"

            resultados.append({
                **config,
                "escalonamento": scaler_nome,
                "label":         label,
                "id":            row_id,
                "Val_Acc":  m_val["Acc"],  "Val_Prec": m_val["Prec"],
                "Val_Sens": m_val["Sens"], "Val_Espec": m_val["Espec"],
                "Val_F1":   m_val["F1"],   "Val_CM":    m_val["cm"],
                "T1_Acc":   m1["Acc"],     "T1_Prec":  m1["Prec"],
                "T1_Sens":  m1["Sens"],    "T1_Espec": m1["Espec"],
                "T1_F1":    m1["F1"],      "T1_CM":    m1["cm"],
                "T2_Acc":   m2["Acc"],     "T2_Prec":  m2["Prec"],
                "T2_Sens":  m2["Sens"],    "T2_Espec": m2["Espec"],
                "T2_F1":    m2["F1"],      "T2_CM":    m2["cm"],
            })

    df = pd.DataFrame(resultados)
    df["Media_F1"] = (2 * df["T1_F1"] * df["T2_F1"]) / (df["T1_F1"] + df["T2_F1"] + 1e-12)
    return df


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

    # (2) Barras — top-N selecionados pela Validação (Val_F1) -----
    top = res_df.sort_values(["Val_F1", "Val_Sens"], ascending=[False, False], kind="stable").head(cfg["top_n_barras"])
    configs_bar = [
        ({"id": r["id"]},
         f"{r['label'].replace(chr(10), ' ')} | {r['escalonamento']}"
         if len(cfg["scalers"]) > 1 else r["label"].replace(chr(10), " "))
        for _, r in top.iterrows()
    ]
    plot_barras(
        df=res_df,
        configs=configs_bar,
        titulo=f"{nome_modelo} — Top {cfg['top_n_barras']} (selecionados por Val_F1)",
        arquivo=os.path.join(PASTA_PLOTS, f"{prefixo}_metricas_comparativo.png"),
    )

    # (3) Matrizes — melhor configuração eleita pela Validação ----
    melhor_val = res_df.sort_values(
        ["Val_F1", "Val_Sens"], ascending=[False, False], kind="stable"
    ).iloc[0]

    config_str = melhor_val["label"].replace(chr(10), " ")
    lbl_modelo = f"{config_str} | {melhor_val['escalonamento']}"
    configs_cm = [({"id": melhor_val["id"]}, lbl_modelo)]

    datasets_cm = [
        ("Val_CM", "Validação - 20%", "validacao", f"Val F1: {melhor_val['Val_F1']:.4f}"),
        ("T1_CM",  "datatest",  "datatest",  f"T1 F1: {melhor_val['T1_F1']:.4f}"),
        ("T2_CM",  "datatest2", "datatest2", f"T2 F1: {melhor_val['T2_F1']:.4f}"),
    ]

    for cm_col, nome_ds, sufixo_arq, f1_txt in datasets_cm:
        plot_matrizes_confusao(
            df=res_df,
            configs=configs_cm,
            cm_col=cm_col,
            prefixos=[("", f"{nome_ds} | {f1_txt}")],
            titulo=f"{nome_modelo} — Matriz de Confusão - {nome_ds}",
            arquivo=os.path.join(PASTA_PLOTS, f"{prefixo}_matriz_{sufixo_arq}.png"),
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
        X_train, y_train = dados["train"]
        X_test1, _ = dados["test1"]
        X_test2, _ = dados["test2"]

        for ds_nome, X in [("datatest", X_test1), ("datatest2", X_test2)]:
            if list(X.columns) != list(X_train.columns):
                raise ValueError(f"{nome}: colunas diferentes em {ds_nome}.")

        corte = int(len(y_train) * 0.80)
        n_exp = len(cfg["configs"]) * len(cfg["scalers"])
        print(f"\n{'='*72}")
        print(f"Modelo: {nome}")
        print(f"{'='*72}")
        print(f"  treino/val: {corte} treino interno (80%) | {len(y_train) - corte} validação (20%)")
        print(f"  features:   {len(X_train.columns)} (formatar={cfg['formatar']})")
        print(f"  configs:    {len(cfg['configs'])} × scalers: {len(cfg['scalers'])} = {n_exp} experimentos")

        res_df = rodar_modelo(nome, cfg, dados)
        todos[nome] = res_df

        # Ordena exclusivamente pela validação (sem tocar em T1/T2)
        res_ordenado = res_df.sort_values(
            ["Val_F1", "Val_Sens"], ascending=[False, False], kind="stable"
        ).reset_index(drop=True)

        if salvar:
            os.makedirs(PASTA_RESULTADOS, exist_ok=True)
            
            # 1) Salva ranking apenas da validação
            cols_val = [c for c in res_ordenado.columns if not c.startswith(("T1_", "T2_", "Media_")) and c not in ("Val_CM", "id")]
            res_ordenado[cols_val].to_csv(
                os.path.join(PASTA_RESULTADOS, f"{nome.lower()}_validacao.csv"),
                index=False,
            )

            # 2) Salva tabela completa ordenada pela validação
            res_ordenado.drop(columns=["Val_CM", "T1_CM", "T2_CM", "id"]).to_csv(
                os.path.join(PASTA_RESULTADOS, f"{nome.lower()}_completo.csv"),
                index=False,
            )

        melhor = res_ordenado.iloc[0]
        print("\n  Melhor configuração na validação:")
        print(f"    {melhor['label'].replace(chr(10), ' ')} | escalonamento={melhor['escalonamento']} | Val_F1={melhor['Val_F1']:.4f}")

        print("\n  Top 5 (selecionados por Val_F1):")
        print(res_ordenado.head(5)[
            ["label", "escalonamento", "Val_F1", "T1_F1", "T2_F1", "Media_F1"]
        ].to_string(index=False, float_format="{:.4f}".format))

        if plots:
            gerar_plots(nome, res_df, cfg)

    return todos


if __name__ == "__main__":
    rodar_todos()
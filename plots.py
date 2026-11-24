"""plots.py — visualizações unificadas para todos os modelos do projeto.

Colunas esperadas em df:
    T1_Acc, T1_Prec, T1_Sens, T1_Espec, T1_F1, T1_CM
    T2_Acc, T2_Prec, T2_Sens, T2_Espec, T2_F1, T2_CM
onde T1 = datatest1, T2 = datatest2.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


METRICAS_NOMES   = ["Acurácia", "Precisão", "Sensibilidade", "Especificidade", "F1-score"]
METRICAS_SUFIXOS = ["Acc", "Prec", "Sens", "Espec", "F1"]
PREFIXOS_PADRAO  = [("T1", "datatest1"), ("T2", "datatest2")]


def _resolver_arquivo(arquivo, grupo=None):
    """arquivo pode ser: None | string (vira {stem}_{grupo}.png) | dict {grupo: caminho}."""
    if arquivo is None:
        return None
    if isinstance(arquivo, dict):
        return arquivo.get(grupo)
    if grupo is not None:
        return arquivo.replace(".png", f"_{grupo}.png")
    return arquivo


def _salvar(fig, caminho):
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=300, bbox_inches="tight")


def plot_curva(
    df, x_col, group_col=None, grupos=None, nomes_grupos=None,
    prefixos=None, titulo="", arquivo=None, xlabel=None,
    marca_x=None, marca_label=None, y_lim=(0.80, 1.01),
):
    """Uma figura por grupo (ex.: 'euclidean', 'manhattan'); dentro da figura,
    um subplot por conjunto de teste (T1, T2)."""
    prefixos = prefixos or PREFIXOS_PADRAO
    nomes_grupos = nomes_grupos or {}
    grupos = grupos if grupos is not None else (
        sorted(df[group_col].unique()) if group_col else [None]
    )

    for grupo in grupos:
        sub = (df[df[group_col] == grupo] if group_col else df).sort_values(x_col)
        x_vals = sub[x_col].tolist()
        pos_x  = np.arange(len(x_vals))

        fig, axes = plt.subplots(1, len(prefixos), figsize=(6.5 * len(prefixos), 5), sharey=True)
        if len(prefixos) == 1:
            axes = [axes]

        for ax, (pref, nome) in zip(axes, prefixos):
            ax.plot(pos_x, sub[f"{pref}_Acc"], marker="o", color="#2ca02c",
                    linewidth=2, label="Acurácia")
            ax.plot(pos_x, sub[f"{pref}_F1"],  marker="s", color="#1f77b4",
                    linestyle="--", linewidth=2, label="F1-score")
            if marca_x is not None and marca_x in x_vals:
                ax.axvline(x_vals.index(marca_x), color="gray", linestyle=":",
                           alpha=0.8, label=marca_label or f"x = {marca_x}")
            ax.set_title(f"{titulo} — {nome}", fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel or x_col)
            ax.set_xticks(pos_x)
            ax.set_xticklabels(x_vals)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=9.5, loc="lower right")

        axes[0].set_ylabel("Pontuação (0 a 1)")
        axes[0].set_ylim(*y_lim)

        nome_grupo = nomes_grupos.get(grupo, grupo) if grupo is not None else None
        sufixo_titulo = f" ({nome_grupo})" if nome_grupo else ""
        plt.suptitle(f"{titulo}{sufixo_titulo}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        _salvar(fig, _resolver_arquivo(arquivo, grupo))
        plt.show()


def plot_matrizes_confusao(
    df, configs, cm_col="CM", prefixos=None,
    titulo="", arquivo=None, cmap="Blues",
):
    """Grade N_configs × N_testes. Cada config é (filtro_dict, label)."""
    prefixos = prefixos or PREFIXOS_PADRAO
    n_cfg, n_tst = len(configs), len(prefixos)

    fig, axes = plt.subplots(n_cfg, n_tst, figsize=(5.5 * n_tst, 4.2 * n_cfg))
    if n_cfg == 1: axes = [axes]
    if n_tst == 1: axes = [[a] for a in axes]

    for row, (filtro, label) in enumerate(configs):
        linha = df.copy()
        for k, v in filtro.items():
            linha = linha[linha[k] == v]
        linha = linha.iloc[0]

        for col, (pref, nome) in enumerate(prefixos):
            col_cm = f"{pref}_{cm_col}" if f"{pref}_{cm_col}" in linha.index else cm_col
            disp = ConfusionMatrixDisplay(linha[col_cm], display_labels=["Vazio (0)", "Ocupado (1)"])
            disp.plot(ax=axes[row][col], cmap=cmap, values_format="d", colorbar=False)
            axes[row][col].set_title(f"{label} — {nome}", fontsize=10.5, fontweight="bold")
            axes[row][col].set_xlabel("Classe Prevista")
            axes[row][col].set_ylabel("Classe Real")

    plt.suptitle(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _salvar(fig, arquivo)
    plt.show()


def plot_barras(
    df, configs, prefixos=None, titulo="", arquivo=None, y_lim=(0.70, 1.06),
):
    """Barras agrupadas: 5 métricas × N configs, um subplot por conjunto de teste."""
    prefixos = prefixos or PREFIXOS_PADRAO
    n = len(prefixos)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), sharey=True)
    if n == 1: axes = [axes]

    x = np.arange(len(METRICAS_NOMES))
    largura = 0.8 / len(configs)

    for ax, (pref, nome) in zip(axes, prefixos):
        for idx, (filtro, label) in enumerate(configs):
            linha = df.copy()
            for k, v in filtro.items():
                linha = linha[linha[k] == v]
            linha = linha.iloc[0]
            valores = [linha[f"{pref}_{s}"] for s in METRICAS_SUFIXOS]
            desloc  = (idx - (len(configs) - 1) / 2) * largura
            bars = ax.bar(x + desloc, valores, largura, label=label)
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{titulo} — {nome}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METRICAS_NOMES, rotation=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Valor da Métrica")
    axes[0].set_ylim(*y_lim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(configs), frameon=True, fontsize=9.5)
    plt.suptitle(titulo, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    _salvar(fig, arquivo)
    plt.show()
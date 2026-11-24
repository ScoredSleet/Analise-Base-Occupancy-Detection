import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Localiza a pasta do repositório para importar preprocessamento.py e ler os CSVs locais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = (
    BASE_DIR
    if os.path.isdir(os.path.join(BASE_DIR, "dataset"))
    else os.path.join(BASE_DIR, "Analise-Base-Occupancy-Detection")
)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from preprocessamento import carregar_base

CSV_DIR = os.path.join(REPO_DIR, "dataset", "occupancy_detection", "dataset_csv")

# 2. Carrega os 3 conjuntos com o pré-processamento padrão do projeto (formatar=True -> 8 atributos)
# Atributos: Temperature, Humidity, Light, CO2, HumidityRatio, date_dia_semana, date_hora_sin, date_hora_cos
X_train, y_train = carregar_base(os.path.join(CSV_DIR, "datatraining.csv"), return_X_y=True, formatar=True)
X_test1, y_test1 = carregar_base(os.path.join(CSV_DIR, "datatest.csv"), return_X_y=True, formatar=True)
X_test2, y_test2 = carregar_base(os.path.join(CSV_DIR, "datatest2.csv"), return_X_y=True, formatar=True)

# 3. Padronização Z-score (ajustada APENAS no treino para evitar data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)
X_test2_scaled = scaler.transform(X_test2)

# Probabilidade a priori (calculada automaticamente pelo GaussianNB no fit):
# P(Occupancy = 0) = 6414 / 8143 = 0.7877 (78,77% sala vazia)
# P(Occupancy = 1) = 1729 / 8143 = 0.2123 (21,23% sala ocupada)


def calcular_metricas(y_true, y_pred):
    """Retorna as 5 métricas definidas na seção 3.3 do relatório e a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    especificidade = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, zero_division=0),
        'Sensibilidade': recall_score(y_true, y_pred, zero_division=0),
        'Especificidade': especificidade,
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cm': cm,
    }


# 5. Experimentos variando apenas o parâmetro var_smoothing do GaussianNB
valores_vs = [1e-9, 1e-3, 1e-2, 0.1, 0.5, 1.0]

resultados = []
for vs in valores_vs:
    nb = GaussianNB(var_smoothing=vs)
    nb.fit(X_train_scaled, y_train)

    m1 = calcular_metricas(y_test1, nb.predict(X_test1_scaled))
    m2 = calcular_metricas(y_test2, nb.predict(X_test2_scaled))

    resultados.append({
        'var_smoothing': vs,
        'T1_Acc': m1['Acurácia'], 'T1_Prec': m1['Precisão'], 'T1_Sens': m1['Sensibilidade'],
        'T1_Espec': m1['Especificidade'], 'T1_F1': m1['F1-score'],
        'T1_TN': m1['tn'], 'T1_FP': m1['fp'], 'T1_FN': m1['fn'], 'T1_TP': m1['tp'], 'T1_CM': m1['cm'],
        'T2_Acc': m2['Acurácia'], 'T2_Prec': m2['Precisão'], 'T2_Sens': m2['Sensibilidade'],
        'T2_Espec': m2['Especificidade'], 'T2_F1': m2['F1-score'],
        'T2_TN': m2['tn'], 'T2_FP': m2['fp'], 'T2_FN': m2['fn'], 'T2_TP': m2['tp'], 'T2_CM': m2['cm'],
    })

res_df = pd.DataFrame(resultados)

# Salva os resultados em CSV na pasta resultados/ (tabela comparativa completa + formato simples_validacao/testes)
RESULTADOS_DIR = os.path.join(REPO_DIR, "resultados")
os.makedirs(RESULTADOS_DIR, exist_ok=True)
res_df.drop(columns=['T1_CM', 'T2_CM']).to_csv(
    os.path.join(RESULTADOS_DIR, "naive_bayes_comparativo_testes.csv"), index=False
)

# Também gera naive_bayes_simples_validacao.csv e naive_bayes_simples_testes.csv (mesmo padrão de svm.py e modelo_xgboost.py)
corte_val = int(len(y_train) * 0.80)
scaler_val = StandardScaler()
X_tr_int = scaler_val.fit_transform(X_train.iloc[:corte_val])
X_val_int = scaler_val.transform(X_train.iloc[corte_val:])
y_tr_int, y_val_int = y_train.iloc[:corte_val], y_train.iloc[corte_val:]

exp_val = []
for vs in valores_vs:
    mv = calcular_metricas(y_val_int, GaussianNB(var_smoothing=vs).fit(X_tr_int, y_tr_int).predict(X_val_int))
    exp_val.append({
        'escalonamento': 'Z-Score', 'var_smoothing': vs,
        'f1': mv['F1-score'], 'acuracia': mv['Acurácia'], 'precisao': mv['Precisão'],
        'sensibilidade': mv['Sensibilidade'], 'especificidade': mv['Especificidade'],
        'tn': mv['tn'], 'fp': mv['fp'], 'fn': mv['fn'], 'tp': mv['tp'],
    })
df_val_nb = pd.DataFrame(exp_val).sort_values(['f1', 'sensibilidade'], ascending=[False, False], kind='stable').reset_index(drop=True)
df_val_nb.to_csv(os.path.join(RESULTADOS_DIR, "naive_bayes_simples_validacao.csv"), index=False)

melhor_nb = df_val_nb.iloc[0]
linha_melhor = res_df[res_df['var_smoothing'] == float(melhor_nb['var_smoothing'])].iloc[0]
pd.DataFrame([
    {
        'conjunto': 'datatest.csv', 'f1': linha_melhor['T1_F1'], 'acuracia': linha_melhor['T1_Acc'],
        'precisao': linha_melhor['T1_Prec'], 'sensibilidade': linha_melhor['T1_Sens'],
        'especificidade': linha_melhor['T1_Espec'], 'tn': int(linha_melhor['T1_TN']),
        'fp': int(linha_melhor['T1_FP']), 'fn': int(linha_melhor['T1_FN']), 'tp': int(linha_melhor['T1_TP']),
    },
    {
        'conjunto': 'datatest2.csv', 'f1': linha_melhor['T2_F1'], 'acuracia': linha_melhor['T2_Acc'],
        'precisao': linha_melhor['T2_Prec'], 'sensibilidade': linha_melhor['T2_Sens'],
        'especificidade': linha_melhor['T2_Espec'], 'tn': int(linha_melhor['T2_TN']),
        'fp': int(linha_melhor['T2_FP']), 'fn': int(linha_melhor['T2_FN']), 'tp': int(linha_melhor['T2_TP']),
    },
]).to_csv(os.path.join(RESULTADOS_DIR, "naive_bayes_simples_testes.csv"), index=False)

print("=== TABELA COMPARATIVA GAUSSIAN NAIVE BAYES (datatest1) ===")
print(res_df[['var_smoothing', 'T1_Acc', 'T1_Prec', 'T1_Sens', 'T1_Espec', 'T1_F1']].to_string(index=False))

print("\n=== TABELA COMPARATIVA GAUSSIAN NAIVE BAYES (datatest2) ===")
print(res_df[['var_smoothing', 'T2_Acc', 'T2_Prec', 'T2_Sens', 'T2_Espec', 'T2_F1']].to_string(index=False))



# GRÁFICO 1: Evolução de Acurácia e F1-score ao variar var_smoothing
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
pos_vs = np.arange(len(valores_vs))
labels_vs = ['1e-9\n(padrão)', '1e-3', '1e-2', '0.1', '0.5', '1.0']

for ax, prefixo, titulo in [
    (axes[0], 'T1', 'datatest1 (Porta Fechada)'),
    (axes[1], 'T2', 'datatest2 (Porta Aberta)'),
]:
    ax.plot(pos_vs, res_df[f'{prefixo}_Acc'], marker='o', color='#2ca02c', linestyle='-', linewidth=2, label='Acurácia')
    ax.plot(pos_vs, res_df[f'{prefixo}_F1'], marker='s', color='#1f77b4', linestyle='--', linewidth=2, label='F1-score')

    ax.set_title(f'GaussianNB — {titulo}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Parâmetro var_smoothing')
    ax.set_xticks(pos_vs)
    ax.set_xticklabels(labels_vs)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9.5, loc='lower left')

axes[0].set_ylabel('Pontuação (0 a 1)')
axes[0].set_ylim(0.80, 1.01)
plt.suptitle('Desempenho do Gaussian Naive Bayes conforme variação de var_smoothing', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('naive_bayes_curva_smoothing.png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTADOS_DIR, 'naive_bayes_curva_smoothing.png'), dpi=300, bbox_inches='tight')


# GRÁFICO 2: Matrizes de Confusão (as 3 configurações comparadas no gráfico de barras: 1e-9, 0.1 e 0.5)
configs_cm = [
    (1e-9, 'GaussianNB (var_smoothing=1e-9, Padrão)'),
    (0.1, 'GaussianNB (var_smoothing=0.1)'),
    (0.5, 'GaussianNB (var_smoothing=0.5)'),
]

fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5))
for row_idx, (vs, nome_cfg) in enumerate(configs_cm):
    linha = res_df[res_df['var_smoothing'] == vs].iloc[0]

    for col_idx, (cm_key, titulo_ds) in enumerate([('T1_CM', 'datatest1'), ('T2_CM', 'datatest2')]):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=linha[cm_key],
            display_labels=['Vazio (0)', 'Ocupado (1)'],
        )
        disp.plot(ax=axes[row_idx, col_idx], cmap='Greens', values_format='d', colorbar=False)
        axes[row_idx, col_idx].set_title(f'{nome_cfg} — {titulo_ds}', fontsize=10.5, fontweight='bold')
        axes[row_idx, col_idx].set_xlabel('Classe Prevista')
        axes[row_idx, col_idx].set_ylabel('Classe Real')

plt.suptitle('Matrizes de Confusão — Comparação de Parametrizações do GaussianNB', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('naive_bayes_matrizes_confusao.png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTADOS_DIR, 'naive_bayes_matrizes_confusao.png'), dpi=300, bbox_inches='tight')


# GRÁFICO 3: Comparativo das 5 Métricas (Barras Agrupadas — 3 Configurações)
configs_bar = [
    (1e-9, 'var_smoothing = 1e-9 (Padrão)'),
    (0.1, 'var_smoothing = 0.1'),
    (0.5, 'var_smoothing = 0.5'),
]
nomes_metricas = ['Acurácia', 'Precisão', 'Sensibilidade', 'Especificidade', 'F1-score']
chaves_metricas = ['Acc', 'Prec', 'Sens', 'Espec', 'F1']

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
x = np.arange(len(nomes_metricas))
largura = 0.25

for ax, prefixo, titulo in [
    (axes[0], 'T1', 'datatest1 (Porta Fechada)'),
    (axes[1], 'T2', 'datatest2 (Porta Aberta)'),
]:
    for idx, (vs, label) in enumerate(configs_bar):
        linha = res_df[res_df['var_smoothing'] == vs].iloc[0]
        valores = [linha[f'{prefixo}_{ch}'] for ch in chaves_metricas]
        barras = ax.bar(x + (idx - 1) * largura, valores, largura, label=label)
        for b in barras:
            h = b.get_height()
            ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5)

    ax.set_title(f'Métricas GaussianNB — {titulo}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes_metricas, rotation=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

axes[0].set_ylabel('Valor da Métrica')
axes[0].set_ylim(0.70, 1.06)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=True, fontsize=9.5)
plt.suptitle('Comparação das 5 Medidas de Avaliação no Gaussian Naive Bayes', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.87])
plt.savefig('naive_bayes_metricas_comparativo.png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTADOS_DIR, 'naive_bayes_metricas_comparativo.png'), dpi=300, bbox_inches='tight')

print("\nGráficos salvos com sucesso:")
print("- naive_bayes_curva_smoothing.png")
print("- naive_bayes_matrizes_confusao.png")
print("- naive_bayes_metricas_comparativo.png")

plt.show()

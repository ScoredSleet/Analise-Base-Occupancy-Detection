import os
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

# 1. Carrega os 3 conjuntos originais diretamente dos CSVs locais (funciona na raiz do repo ou na pasta pai)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "dataset", "occupancy_detection", "dataset_csv")
if not os.path.isdir(CSV_DIR):
    CSV_DIR = os.path.join(
        BASE_DIR, "Analise-Base-Occupancy-Detection", "dataset", "occupancy_detection", "dataset_csv"
    )

train_df = pd.read_csv(os.path.join(CSV_DIR, "datatraining.csv"))  # datatraining (8.143 linhas)
test1_df = pd.read_csv(os.path.join(CSV_DIR, "datatest.csv"))      # datatest1    (2.665 linhas)
test2_df = pd.read_csv(os.path.join(CSV_DIR, "datatest2.csv"))     # datatest2    (9.752 linhas)

# 3. Seleciona os 5 atributos numéricos contínuos (sem 'date') e converte para float / int
colunas_X = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']

X_train = train_df[colunas_X].astype(float)
y_train = train_df['Occupancy'].astype(int)

X_test1 = test1_df[colunas_X].astype(float)
y_test1 = test1_df['Occupancy'].astype(int)

X_test2 = test2_df[colunas_X].astype(float)
y_test2 = test2_df['Occupancy'].astype(int)

# 4. Padronização Z-score (ajustada APENAS no treino para evitar data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)
X_test2_scaled = scaler.transform(X_test2)

# Probabilidade a priori (calculada automaticamente pelo GaussianNB no fit):
# P(Occupancy = 0) = 6414 / 8143 = 0.7877 (78,77% sala vazia)
# P(Occupancy = 1) = 1729 / 8143 = 0.2123 (21,23% sala ocupada)


def calcular_metricas(y_true, y_pred):
    """Retorna as 5 métricas definidas na seção 3.3 do relatório e a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    especificidade = tn / (tn + fp)
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred),
        'Sensibilidade': recall_score(y_true, y_pred),
        'Especificidade': especificidade,
        'F1-score': f1_score(y_true, y_pred),
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
        'T1_Espec': m1['Especificidade'], 'T1_F1': m1['F1-score'], 'T1_CM': m1['cm'],
        'T2_Acc': m2['Acurácia'], 'T2_Prec': m2['Precisão'], 'T2_Sens': m2['Sensibilidade'],
        'T2_Espec': m2['Especificidade'], 'T2_F1': m2['F1-score'], 'T2_CM': m2['cm'],
    })

res_df = pd.DataFrame(resultados)

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

print("\nGráficos salvos com sucesso:")
print("- naive_bayes_curva_smoothing.png")
print("- naive_bayes_matrizes_confusao.png")
print("- naive_bayes_metricas_comparativo.png")

plt.show()

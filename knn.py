import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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


# 5. Experimentos variando K e métrica de distância (Euclidiana vs Manhattan)
valores_k = [1, 3, 5, 11, 21, 51, 91, 151]
metricas_dist = ['euclidean', 'manhattan']

resultados = []
for metric in metricas_dist:
    for k in valores_k:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train_scaled, y_train)

        m1 = calcular_metricas(y_test1, knn.predict(X_test1_scaled))
        m2 = calcular_metricas(y_test2, knn.predict(X_test2_scaled))

        resultados.append({
            'Metrica': metric,
            'k': k,
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
    os.path.join(RESULTADOS_DIR, "knn_comparativo_testes.csv"), index=False
)

# Também gera knn_simples_validacao.csv e knn_simples_testes.csv (mesmo padrão de svm.py e modelo_xgboost.py)
corte_val = int(len(y_train) * 0.80)
scaler_val = StandardScaler()
X_tr_int = scaler_val.fit_transform(X_train.iloc[:corte_val])
X_val_int = scaler_val.transform(X_train.iloc[corte_val:])
y_tr_int, y_val_int = y_train.iloc[:corte_val], y_train.iloc[corte_val:]

exp_val = []
for metric in metricas_dist:
    for k in valores_k:
        mv = calcular_metricas(y_val_int, KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_tr_int, y_tr_int).predict(X_val_int))
        exp_val.append({
            'escalonamento': 'Z-Score', 'metric': metric, 'n_neighbors': k,
            'f1': mv['F1-score'], 'acuracia': mv['Acurácia'], 'precisao': mv['Precisão'],
            'sensibilidade': mv['Sensibilidade'], 'especificidade': mv['Especificidade'],
            'tn': mv['tn'], 'fp': mv['fp'], 'fn': mv['fn'], 'tp': mv['tp'],
        })
df_val_knn = pd.DataFrame(exp_val).sort_values(['f1', 'sensibilidade'], ascending=[False, False], kind='stable').reset_index(drop=True)
df_val_knn.to_csv(os.path.join(RESULTADOS_DIR, "knn_simples_validacao.csv"), index=False)

melhor_knn = df_val_knn.iloc[0]
linha_melhor = res_df[(res_df['Metrica'] == melhor_knn['metric']) & (res_df['k'] == int(melhor_knn['n_neighbors']))].iloc[0]
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
]).to_csv(os.path.join(RESULTADOS_DIR, "knn_simples_testes.csv"), index=False)

print("=== TABELA COMPARATIVA KNN (datatest1) ===")
print(res_df[['Metrica', 'k', 'T1_Acc', 'T1_Prec', 'T1_Sens', 'T1_Espec', 'T1_F1']].to_string(index=False, float_format='{:.4f}'.format))

print("\n=== TABELA COMPARATIVA KNN (datatest2) ===")
print(res_df[['Metrica', 'k', 'T2_Acc', 'T2_Prec', 'T2_Sens', 'T2_Espec', 'T2_F1']].to_string(index=False, float_format='{:.4f}'.format))


# GRÁFICOS 1A e 1B: Curvas separadas para Euclidiana e Manhattan (mesmo padrão do Naive Bayes)
pos_k = np.arange(len(valores_k))
idx_91 = valores_k.index(91)

for metric, nome_dist, arquivo_saida in [
    ('euclidean', 'Distância Euclidiana', 'knn_curva_k_euclidiana.png'),
    ('manhattan', 'Distância Manhattan', 'knn_curva_k_manhattan.png'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sub = res_df[res_df['Metrica'] == metric]

    for ax, prefixo, titulo in [
        (axes[0], 'T1', 'datatest1 (Porta Fechada)'),
        (axes[1], 'T2', 'datatest2 (Porta Aberta)'),
    ]:
        ax.plot(pos_k, sub[f'{prefixo}_Acc'], marker='o', color='#2ca02c', linestyle='-', linewidth=2, label='Acurácia')
        ax.plot(pos_k, sub[f'{prefixo}_F1'], marker='s', color='#1f77b4', linestyle='--', linewidth=2, label='F1-score')
        ax.axvline(idx_91, color='gray', linestyle=':', alpha=0.8, label='k = 91 (≈ √N)')

        ax.set_title(f'KNN ({nome_dist}) — {titulo}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Número de vizinhos (k)')
        ax.set_xticks(pos_k)
        ax.set_xticklabels(valores_k)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=9.5, loc='lower right')

    axes[0].set_ylabel('Pontuação (0 a 1)')
    axes[0].set_ylim(0.80, 1.01)
    plt.suptitle(f'Desempenho do KNN ({nome_dist}) conforme variação de k', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTADOS_DIR, arquivo_saida), dpi=300, bbox_inches='tight')
    if metric == 'euclidean':
        plt.savefig('knn_curva_k.png', dpi=300, bbox_inches='tight')



# GRÁFICOS 2A e 2B: Matrizes de Confusão separadas para Euclidiana e Manhattan (k=5 vs k=91)
for metric, nome_dist, cmap_cor, arquivo_cm in [
    ('euclidean', 'Euclidiana', 'Blues', 'knn_matrizes_confusao_euclidiana.png'),
    ('manhattan', 'Manhattan', 'Blues', 'knn_matrizes_confusao_manhattan.png'),
]:
    configs_cm = [
        (metric, 5, f'KNN (k=5, {nome_dist})'),
        (metric, 91, f'KNN (k=91, {nome_dist})'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for row_idx, (m_key, k, nome_cfg) in enumerate(configs_cm):
        linha = res_df[(res_df['Metrica'] == m_key) & (res_df['k'] == k)].iloc[0]

        for col_idx, (cm_key, titulo_ds) in enumerate([('T1_CM', 'datatest1'), ('T2_CM', 'datatest2')]):
            disp = ConfusionMatrixDisplay(
                confusion_matrix=linha[cm_key],
                display_labels=['Vazio (0)', 'Ocupado (1)'],
            )
            disp.plot(ax=axes[row_idx, col_idx], cmap=cmap_cor, values_format='d', colorbar=False)
            axes[row_idx, col_idx].set_title(f'{nome_cfg} — {titulo_ds}', fontsize=11, fontweight='bold')
            axes[row_idx, col_idx].set_xlabel('Classe Prevista')
            axes[row_idx, col_idx].set_ylabel('Classe Real')

    plt.suptitle(f'Matrizes de Confusão — KNN (Distância {nome_dist})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(arquivo_cm, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTADOS_DIR, arquivo_cm), dpi=300, bbox_inches='tight')
    if metric == 'euclidean':
        plt.savefig('knn_matrizes_confusao.png', dpi=300, bbox_inches='tight')



# GRÁFICO 3: Comparativo das 5 Métricas (Barras Agrupadas — 4 Configurações)
configs_bar = [
    ('euclidean', 5, 'k=5 (Euclidiana)'),
    ('manhattan', 5, 'k=5 (Manhattan)'),
    ('euclidean', 91, 'k=91 (Euclidiana)'),
    ('manhattan', 91, 'k=91 (Manhattan)'),
]
nomes_metricas = ['Acurácia', 'Precisão', 'Sensibilidade', 'Especificidade', 'F1-score']
chaves_metricas = ['Acc', 'Prec', 'Sens', 'Espec', 'F1']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
x = np.arange(len(nomes_metricas))
largura = 0.20

for ax, prefixo, titulo in [
    (axes[0], 'T1', 'datatest1 (Porta Fechada)'),
    (axes[1], 'T2', 'datatest2 (Porta Aberta)'),
]:
    for idx, (metric, k, label) in enumerate(configs_bar):
        linha = res_df[(res_df['Metrica'] == metric) & (res_df['k'] == k)].iloc[0]
        valores = [linha[f'{prefixo}_{ch}'] for ch in chaves_metricas]
        deslocamento = (idx - 1.5) * largura
        barras = ax.bar(x + deslocamento, valores, largura, label=label)
        for b in barras:
            h = b.get_height()
            ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    ax.set_title(f'Métricas KNN — {titulo}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes_metricas, rotation=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

axes[0].set_ylabel('Valor da Métrica')
axes[0].set_ylim(0.70, 1.06)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=True, fontsize=9.5)
plt.suptitle('Comparação das 5 Medidas de Avaliação no KNN', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.87])
plt.savefig('knn_metricas_comparativo.png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTADOS_DIR, 'knn_metricas_comparativo.png'), dpi=300, bbox_inches='tight')

print("\nGráficos salvos com sucesso:")
print("- knn_curva_k_euclidiana.png")
print("- knn_curva_k_manhattan.png")
print("- knn_matrizes_confusao_euclidiana.png")
print("- knn_matrizes_confusao_manhattan.png")
print("- knn_metricas_comparativo.png")

plt.show()

# Faixas válidas por coluna
# date          - (timestamp)
# Temperature   - [-40, 80] °C
# Humidity      - [0, 100] % RH
# Light         - [1, 40000] lux
# CO2           - [0, 2000] ppm
# HumidityRatio - derivada (sem faixa própria)
# Occupancy     - {0, 1}

# Como temos esses dados ficam mais faceis de trabalhar nas colunas

import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def reducao(dataset: DataFrame, colunas: list[str]) -> DataFrame:
    d = dataset.drop(colunas, axis=1)
    
    return d

def formatar_data(dataset: pd.DataFrame, coluna: str = 'date', drop_original: bool = True) -> pd.DataFrame:
    df = dataset.copy()
    
    df[coluna] = pd.to_datetime(df[coluna], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(coluna).reset_index(drop=True)
    
    df[f'{coluna}_dia_semana'] = np.where(df[coluna].dt.weekday >= 5, 1, 0)

    hora = df[coluna].dt.hour
    df[f'{coluna}_hora_sin'] = np.sin(2 * np.pi * hora / 24.0)
    df[f'{coluna}_hora_cos'] = np.cos(2 * np.pi * hora / 24.0)

    if drop_original:
        df = df.drop(columns=[coluna])
        
    return df

def carregar_base(path, return_X_y: bool = False, formatar: bool = False):
    data = pd.read_csv(path)
    
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values('date').reset_index(drop=True)
    
    if formatar:
        data = formatar_data(data, 'date')
    else:
        data = reducao(data, ['date'])
        
    
    if return_X_y:
        X = data.drop(columns=['Occupancy'])
        y = data['Occupancy']
        return X, y
        
    return data
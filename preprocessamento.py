# Faixas válidas por coluna
# date          - (timestamp)
# Temperature   - [-40, 80] °C
# Humidity      - [0, 100] % RH
# Light         - [1, 40000] lux
# CO2           - [0, 2000] ppm
# HumidityRatio - derivada (sem faixa própria)
# Occupancy     - {0, 1}

# Como temos esses dados ficam mais faceis de trabalhar nas colunas

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

def reducao(dataset: DataFrame, colunas: tuple[str]) -> DataFrame:
    d = dataset.drop(columns=colunas, axis=1)
    
    return d

def min_max(dataset: DataFrame, colunas: tuple[str]):
    df = dataset.copy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = dataset.copy()
    df[list(colunas)] = scaler.fit_transform(df[list(colunas)])

    return df, scaler

def z_score(dataset: DataFrame, colunas: tuple[str]):
    df = dataset.copy()
        
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = dataset.copy()
    df[list(colunas)] = scaler.fit_transform(df[list(colunas)])

    return df, scaler

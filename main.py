'''
Atribuições do projeto e base de Dados

- https://archive.ics.uci.edu/dataset/357/occupancy+detection?utm_source=chatgpt.com
    - Base de dados sobre detecção de ocupação
    - Artigo em anexo que elucida bem o que é
        Olhar EDA antes qualquer coisa
'''
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

X, y = load_wine(return_X_y=True)

models = {"KNN": KNeighborsClassifier(), "Random forest": RandomForestClassifier(random_state=0)}
scalers = {"no scaling": None, "MinMaxScaler": MinMaxScaler(), "StandardScaler": StandardScaler()}

for model_name, model in models.items():
    for scaler_name, scaler in scalers.items():
        pipe = make_pipeline(scaler, model) if scaler else model
        acc = cross_val_score(pipe, X, y, cv=5).mean()
        print(f"{model_name:>13} | {scaler_name:<14} | accuracy {acc:.6f}")
        
salary = np.array([[3.1], [3.4], [3.8], [4.0], [4.2], [4.5], [5.0], [60.0]])

for scaler in [MinMaxScaler(), StandardScaler(), RobustScaler()]:
    out = scaler.fit_transform(salary).ravel()
    print(f"{type(scaler).__name__:>14}: {np.round(out, 2)}")
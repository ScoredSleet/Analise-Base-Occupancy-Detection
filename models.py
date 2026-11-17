# Classe presente os modelos apresentados no trabalho

from sklearn.neighbors import KNeighborsClassifier

class OccupancyKNN(KNeighborsClassifier):
    
    def __init__(self, n_neighbors: int = 5, weights='uniform', algorithm='auto', p=2, metric='minkowski'):
        
        super().__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p, metric=metric)
        
    def fit(self, dataset):
        
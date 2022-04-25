from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
    CondensedNearestNeighbour,
    TomekLinks,
    ClusterCentroids
)

class UndersampleMethod:
    RANDOM = RandomUnderSampler(random_state=42)
    CLUSTER = ClusterCentroids(random_state=42)
    NEARMISS = NearMiss(version=3, n_neighbors=3)
    CONDENSED = CondensedNearestNeighbour(n_neighbors=1)
    TOMEK = TomekLinks()

def undersample_with(strategy, X, y):
    return strategy.fit_resample(X, y)

#
# This code is intentionally missing!
# Read the directions on the course lab page!
#%%
import pandas as pd
import numpy as np

X = pd.read_csv(r"C:\Users\Sjaak\Documents\DAT210x\Module6\Datasets\parkinsons.data", header=0)
X = X.drop('name', axis=1)

y = X['status']
X = X.drop('status', axis=1)

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

#%%
from sklearn.svm import SVC

svc = SVC()

#%%
svc.fit(X_train, y_train)

#%%
score = svc.score(X_test, y_test)
print('score: %.10f' % score)

#%%
best_score = 0
for c in np.arange(0.05, 2, .05):
    for gamma in np.arange(0.001, .1, .001):
        svc = SVC(C=c, gamma=gamma)
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        best_score = score if score > best_score else best_score

print('best_score for %.8f' % best_score)

#%%
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler

for scaler in [Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    preprocessor = scaler()
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    best_score = 0
    for c in np.arange(0.05, 2.05, .05):
            for gamma in np.arange(0.001, .1001, .001):
                svc = SVC(C=c, gamma=gamma)
                svc.fit(X_train, y_train)
                score = svc.score(X_test, y_test)
                best_score = score if score > best_score else best_score

    print('best_score for %s: %.8f' % (scaler.__name__, best_score))

#%%
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler
from sklearn.decomposition import PCA

for scaler in [Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler]:
    for k in range(4, 15):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
        preprocessor = scaler()
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        pca = PCA(n_components=k)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        best_score = 0
        for c in np.arange(0.05, 2.05, .05):
                for gamma in np.arange(0.001, .1001, .001):
                    svc = SVC(C=c, gamma=gamma)
                    svc.fit(X_train, y_train)
                    score = svc.score(X_test, y_test)
                    best_score = score if score > best_score else best_score
    
        print('best_score for %s: %.8f' % (scaler.__name__, best_score))

#%%
from sklearn.manifold import Isomap

for scaler in [Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler]:
    for k in range(2,6):
        for j in range(4,7):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
            preprocessor = scaler()
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)
            iso = Isomap(n_neighbors=k, n_components=j)
            iso.fit(X_train)
            X_train = iso.transform(X_train)
            X_test = iso.transform(X_test)
            best_score = 0
            for c in np.arange(0.05, 2.05, .05):
                    for gamma in np.arange(0.001, .1001, .001):
                        svc = SVC(C=c, gamma=gamma)
                        svc.fit(X_train, y_train)
                        score = svc.score(X_test, y_test)
                        best_score = score if score > best_score else best_score

    print('best_score for %s: %.8f' % (scaler.__name__, best_score))
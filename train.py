import pandas as pd
import numpy as np
# from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    data = pd.read_csv('encodings.csv', index_col=[0])

    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

    # for i in range(1,500):
    #     model = KNeighborsClassifier(probability=True, kernel='poly', degree=i)
    #     model.fit(x_train, y_train)
    #     score = accuracy_score(y_test, model.predict(x_test))
    #     probs = model.predict_proba(x_test)
    
    #     arr_max = []
    #     for arr in probs:
    #         arr_max.append(np.amax(arr))
        
    #     if len(arr_max) == (pd.Series(arr_max) > 0.6).sum() and score > 0.9:
    #         print('model trained1...\n')
    #         return model
    # print('model trained2...\n')
    # return model

    def get_best_k(x_train, y_train, x_test, y_test):
        k = 0
        max_score = 0
        for i in range(5,len(x_train)):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(x_train, y_train)
            score = accuracy_score(y_test, model.predict(x_test))
            if score > max_score:
                max_score, k = score, i
            if score == 1:
                break
        return k

    k = get_best_k(x_train, y_train, x_test, y_test)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    score = accuracy_score(y_train, model.predict(x_train))
    print(f'training score: {score}')

    return model

model = train_model()
joblib.dump(model, 'trained_models/facerec_model.model')
print('model saved...')
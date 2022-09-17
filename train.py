import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    data = pd.read_csv('encodings.csv', index_col=[0])

    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

    for i in range(1,100):
        model = SVC(probability=True, kernel='poly', degree=i)
        model.fit(x_train, y_train)
        score = accuracy_score(y_test, model.predict(x_test))
        probs = model.predict_proba(x_test)
    
        arr_max = []
        for arr in probs:
            arr_max.append(np.amax(arr))
        
        if len(arr_max) == (pd.Series(arr_max) > 0.6).sum() and score == 1:
            print('model trained...\n')
            return model
    print('model saved...\n')
    return model

model = train_model()
joblib.dump(model, 'trained_models/facerec_model.model')
print('model saved...')
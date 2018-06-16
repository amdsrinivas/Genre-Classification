
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.externals import joblib


# In[2]:


level0_model=joblib.load("svm_model_level0")
level1A_model=joblib.load("svm_model_level1A")
level1B_model=joblib.load("svm_model_level1B")
level2AA_model=joblib.load("svm_model_level2AA")
level2AB_model=joblib.load("svm_model_level2AB")
level2BA_model=joblib.load("svm_model_level2BA")
level2BB_model=joblib.load("svm_model_level2BB")


# In[3]:


def ensemble_predict(y):
    level0_predict=level0_model.predict(y)
    classA,classB=level0_model.classes_
    for level0_class in level0_predict:
        if level0_class==classA:
            level1_predict=level1A_model.predict(y)
            classAA,classAB=level1A_model.classes_
            for level1_class in level1_predict:
                if level1_class==classAA:
                    level2_predict=level2AA_model.predict(y)
                else:
                    level2_predict=level2AB_model.predict(y)              
        else:
            level1_predict=level1B_model.predict(y)
            classBA,classBB=level1B_model.classes_
            for level1_class in level1_predict:
                if level1_class==classBA:
                    level2_predict=level2BA_model.predict(y)
                else:
                    level2_predict=level2BB_model.predict(y)
    return level2_predict
def predict(Y):
    predictions=[]
    for y in Y:
        predictions.extend(ensemble_predict([y]))
    return predictions


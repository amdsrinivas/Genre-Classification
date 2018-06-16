
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from copy import deepcopy


# In[2]:


dataset = pd.read_csv("FMA_8000_with_track_and_genres.csv")


# In[3]:


All_class=["Electronic","Experimental","Hip-Hop","Instrumental","Folk","International","Pop","Rock"]
class_listA=["Electronic","Experimental","Hip-Hop","Instrumental"]
class_listB=["Folk","International","Pop","Rock"]
#class_listA=["Electronic","Folk","Hip-Hop","Rock"]
#class_listB=["Instrumental","International","Experimental","Pop"]

class_listAA=class_listA[:2]
class_listAB=class_listA[2:]
class_listBA=class_listB[:2]
class_listBB=class_listB[2:]

def mask_genres(dataframe,class_list):
    new_data_frame=deepcopy(dataframe)
    mask=[x for x in All_class if x not in class_list]
    for x in mask:
        new_data_frame=new_data_frame[new_data_frame.genre!=x]
    return new_data_frame
def convert_to_class(x,class_list_A,class_list_B):
    if x in class_list_A:
        return ",".join(class_list_A)
    elif x in class_list_B:
        return ",".join(class_list_B)
    else:
        return "Invalid"
def filter_invalid(dataset,col):
    new_dataset=deepcopy(dataset)
    new_dataset=new_dataset[new_dataset[col]!="Invalid"]
    return new_dataset


# In[4]:


hirerarchial_dataset=deepcopy(dataset)
hirerarchial_dataset=hirerarchial_dataset.drop("Unnamed: 0",axis=1)
hirerarchial_dataset["genre_level0"]=dataset["genre"].apply(lambda x:convert_to_class(x,class_listA,class_listB))
hirerarchial_dataset["genre_level1_A"]=dataset["genre"].apply(lambda x:convert_to_class(x,class_listAA,class_listAB))
hirerarchial_dataset["genre_level1_B"]=dataset["genre"].apply(lambda x:convert_to_class(x,class_listBA,class_listBB))
hirerarchial_dataset["genre_level2_AA"]=dataset["genre"].apply(lambda x:convert_to_class(x,[class_listAA[0]],[class_listAA[1]]))
hirerarchial_dataset["genre_level2_AB"]=dataset["genre"].apply(lambda x:convert_to_class(x,[class_listAB[0]],[class_listAB[1]]))
hirerarchial_dataset["genre_level2_BA"]=dataset["genre"].apply(lambda x:convert_to_class(x,[class_listBA[0]],[class_listBA[1]]))
hirerarchial_dataset["genre_level2_BB"]=dataset["genre"].apply(lambda x:convert_to_class(x,[class_listBB[0]],[class_listBB[1]]))
hirerarchial_dataset


# Level 0:-

# In[5]:


test_size=.30



# Finally:-

# In[6]:


cols=hirerarchial_dataset.iloc[:,160:].columns
base_level=filter_invalid(hirerarchial_dataset,"genre")
features=base_level.drop(cols,axis=1).as_matrix()     
lables=base_level.genre.as_matrix()
scaled_features=preprocessing.scale(features,axis=0)
#scaled_features = features
scaled_features.shape


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features, lables, test_size=test_size, random_state=42,stratify=lables)


# In[8]:


import model


# In[9]:


print("Ensemble statistics :-")
print("Training accuracy : ",accuracy_score(y_train,model.predict(X_train)))
print("Testing accuracy : ",accuracy_score(y_test,model.predict(X_test)))
print("Confusion metrics for testing :-\n",confusion_matrix(y_test,model.predict(X_test),labels=All_class))
print("Genres considered : ",",".join(All_class))


# In[10]:


from pandas_ml import ConfusionMatrix


# In[12]:
print("Confusion metrics for testing :-\n")
cm=ConfusionMatrix(y_test,model.predict(X_test))
cm.print_stats()

#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier #MachineLearning-Bibliothek
from sklearn.model_selection import train_test_split
from sklearn import tree

music_data = pd.read_csv("C:/Users/tayla/OneDrive/Desktop/Data Science/music.csv")

X = music_data.drop(columns = ["genre"]) #Konvention das X (input) eine Testsatz darstellt
y = music_data["genre"] #(output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #Teilung in Trainings - und Testdaten; 0,2 = 20 % der Gesamtdaten

model = DecisionTreeClassifier()
model.fit(X_train,y_train) #Input "X" und Output "y" Datensatz -> Training des Models
                #Anstatt nur X (input) und y (output) nun tatsächlich X_train und y_train

    
predictions = model.predict(X_test) # [ [21, 1 ], [22, 0] ] -> Alter und 1 = Mann; 0 = Frau


score = accuracy_score(y_test, predictions) #Genauigkeit zwischen 0 - 1, Herannahme von Randomdaten aus der Datenmenge
#score

tree.export_graphviz(model, out_file = "music-recommender.dot",
                                        feature_names = ["age", "gender"], 
                                        class_names = sorted(y.unique()),
                                        label = "all",
                                        rounded = True, #Runde Ecken 
                                        filled = True) #Farbfüllung


# In[ ]:





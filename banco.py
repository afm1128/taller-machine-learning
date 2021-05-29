# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:08:05 2021

@author: Verde
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.metrics import classification_report, roc_curve, precision_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

#Importamos la Data
url = ''
data = pd.read_csv(url)

#ahora remplazaremos todos los valores que hay por datos numericos
data.job.replace(['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 
                  'self-employed', 'unemployed', 'housemaid', 'student'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)

data.education.replace(['tertiary', 'secondary', 'unknown', 'primary'], [0, 1, 2, 3], inplace=True)

data.default.replace(['no', 'yes'], [0, 1], inplace=True)

data.housing.replace(['no', 'yes'], [0, 1], inplace=True)

data.loan.replace(['no', 'yes'], [0, 1], inplace=True)

data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace=True)

data.month.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

data.poutcome.replace(['failure', 'success', 'other', 'unknown'], [0, 1, 2, 3], inplace=True)

data.y.replace(['no', 'yes'], [0, 1], inplace=True)

#confirmamos que todo este correcto
data.info()

##Empezamos hacer los rangos para grupar la data
# Definimos los rangos de balance
rangosBalance = [-8020, -2000, 0, 1000, 2000, 4000, 10000, 20000, 105000]
nombresBalance = ['1', '2', '3', '4', '5', '6', '7', '8']
data.balance = pd.cut(data.balance, rangosBalance, labels=nombresBalance)

#Definimos los rangos de campaña
rangosCampaña = [0, 1, 2, 3, 4, 5, 10, 70]
nombresCampaña = ['1', '2', '3', '4', '5', '6', '7']
data.campaign = pd.cut(data.campaign, rangosCampaña, labels=nombresCampaña)

#Definimos los rangos dia
rangosDia = [0, 5, 10, 15, 20, 25, 30, 40]
nombresDia = ['1', '2', '3', '4', '5', '6', '7']
data.day = pd.cut(data.day, rangosDia, labels=nombresDia)

#Definimos los rangos de duracion
rangosDuracion = [-1, 50, 100, 150, 200, 250, 400, 600, 1000, 5000]
nombresDuracion = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
data.duration = pd.cut(data.duration, rangosDuracion, labels=nombresDuracion)

#Definimos los rangos  de edad
rangosEdad = [10, 25, 30, 40, 50, 60, 100]
nombresEdad = ['1', '2', '3', '4', '5', '6']
data.age = pd.cut(data.age, rangosEdad, labels=nombresEdad)

#Definimos los rangos de PDays
rangosPDays = [-2, -1, 50, 100, 150, 250, 400, 900]
nombresPDays = ['1', '2', '3', '4', '5', '6', '7']
data.pdays = pd.cut(data.pdays, rangosPDays, labels=nombresPDays)

#Definimos los rangos de previo
rangosPrevio = [-1, 0, 10, 300]
nombresPrevio = ['1', '2', '3']
data.previous = pd.cut(data.previous, rangosPrevio, labels=nombresPrevio)


## 2. Utilizar Cross validation para entrenar y probar los modelos con mínimo 10 splits
#Definimos el método de entrenamiento.
def entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred


#Definimos el método matriz de confusión
def matriz_confusion_auc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr

def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix,);
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

def plot_roc_curve(fpr, tpr, label):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


##Entrenamiento y validación cruzada
# 0 = No y 1 = Si
x = np.array(data.drop(['y'], 1))
y = np.array(data.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

## 1. Entrenar 5 modelos con distintos algoritmos de Machine Learning
#Modelo Regresión Logistica
modelRL, acc_validationRL, acc_testRL, y_predRL = entrenamiento(LogisticRegression(), x_train, x_test, y_train, y_test)
matriz_confusionRL, AUCRL, fprRL, tprRL = matriz_confusion_auc(modelRL, x_test, y_test, y_predRL)

#Modelo Gausiano
modelGA, acc_validationGA, acc_testGA, y_predGA = entrenamiento(GaussianNB(), x_train, x_test, y_train, y_test)
matriz_confusionGA, AUCGA, fprGA, tprGA = matriz_confusion_auc(modelGA, x_test, y_test, y_predGA)

#Modelo Perceptrón Multicapa
modelPM, acc_validationPM, acc_testPM, y_predPM = entrenamiento(MLPClassifier(), x_train, x_test, y_train, y_test)
matriz_confusionPM, AUCPM, fprPM, tprPM = matriz_confusion_auc(modelPM, x_test, y_test, y_predPM)

#Modelo Vecinos Cercanos
modelVC, acc_validationVC, acc_testVC, y_predVC = entrenamiento(KNeighborsClassifier(), x_train, x_test, y_train, y_test)
matriz_confusionVC, AUCVC, fprVC, tprVC = matriz_confusion_auc(modelVC, x_test, y_test, y_predVC)

#Modelo Arbol de Decisión
modelAD, acc_validationAD, acc_testAD, y_predAD = entrenamiento(DecisionTreeClassifier(), x_train, x_test, y_train, y_test)
matriz_confusionAD, AUCAD, fprAD, tprAD = matriz_confusion_auc(modelAD, x_test, y_test, y_predAD)


## 3. Realizar usando la librería Pandas una tabla que resuma los resultados de las métricas de cada modelo. En las filas debe ir el nombre del
## modelo empleado y en las columnas los siguientes datos (ordenar de mayor a menor por la métrica AUC):
## a. Accuracy de Entrenamiento
## b. Accuracy de Validación
## c. Accuracy de Test
## d. Recall del Modelo
## e. Precisión del Modelo
## f. F1-Score del Modelo
## g. Área bajo la Curva (AUC)
table = {'Métricas': ['Regresión Logística', 'Gausiano', 'Perceptrón Multicapa', 'Vecinos Cercanos', 'Arbol de Decision'],
         'Accuracy de Entreno': [round(accuracy_score(y_test, y_predRL), 2), round(accuracy_score(y_test, y_predGA), 2),
                                 round(accuracy_score(y_test, y_predPM), 2), round(accuracy_score(y_test, y_predVC), 2),
                                 round(accuracy_score(y_test, y_predAD), 2)],
         'Accuracy de Validación': [round(acc_validationRL, 2), round(acc_validationGA, 2), round(acc_validationPM, 2),
                                    round(acc_validationVC, 2), round(acc_validationAD, 2)],
         'Accuracy de Test': [round(acc_testRL, 2), round(acc_testGA, 2), round(acc_testPM, 2), round(acc_testVC, 2),
                              round(acc_testAD, 2)],
         'Recall': [round(recall_score(y_test, y_predRL), 2), round(recall_score(y_test, y_predGA), 2), 
                    round(recall_score(y_test, y_predPM), 2), round(recall_score(y_test, y_predVC), 2),
                    round(recall_score(y_test, y_predAD), 2)],
         'Precision': [round(precision_score(y_test, y_predRL), 2), round(precision_score(y_test, y_predGA), 2),
                       round(precision_score(y_test, y_predPM), 2), round(precision_score(y_test, y_predVC), 2),
                       round(precision_score(y_test, y_predAD), 2)],
         'F1-Score': [round(f1_score(y_test, y_predRL), 2), round(f1_score(y_test, y_predGA), 2), round(f1_score(y_test, y_predPM), 2),
                      round(f1_score(y_test, y_predVC), 2), round(f1_score(y_test, y_predAD), 2)],
         'AUC': [round(AUCRL, 2), round(AUCGA, 2), round(AUCPM, 2), round(AUCVC, 2), round(AUCAD, 2)]}

print(pd.DataFrame(table))


## 4. Imprimir la matriz de confusión de cada modelo usando la librería pandas
#Regresión logistica
print("Matriz de confusión regresión logistica:")
print(confusion_matrix( y_test, y_predRL))
mostrar_resultados(y_test, y_predRL)
#Gausiano
print("Matriz de confusión gausiano:")
print(confusion_matrix( y_test, y_predGA))
mostrar_resultados(y_test, y_predGA)
#Perceptrón Multicapa
print("Matriz de confusión perceptrón multicapa:")
print(confusion_matrix( y_test, y_predPM))
mostrar_resultados(y_test, y_predPM)
#Vecinos Cercanos
print("Matriz de confusión vecinos cercanos:")
print(confusion_matrix( y_test, y_predVC))
mostrar_resultados(y_test, y_predVC)
#Árbol de Decisión
print("Matriz de confusión árbol de decisión:")
print(confusion_matrix( y_test, y_predAD))
mostrar_resultados(y_test, y_predAD)


# 5. Imprimir las 5 matrices de confusión en un solo gráfico, empleando el mapa de calor de la librería Seaborn
fig = plt.figure(figsize=(11, 15))

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)

sns.heatmap(matriz_confusionRL, ax=ax1, cmap="YlOrRd", annot=True)
sns.heatmap(matriz_confusionGA, ax=ax2, cmap="YlOrRd", annot=True)
sns.heatmap(matriz_confusionPM, ax=ax3, cmap="YlOrRd", annot=True)
sns.heatmap(matriz_confusionVC, ax=ax4, cmap="YlOrRd", annot=True)
sns.heatmap(matriz_confusionAD, ax=ax5, cmap="YlOrRd", annot=True)
plt.show()


##6. Imprimir las métricas de precisión, recall y f1 de cada clase de cada modelo
print("Métricas regresión logistica: ")
mostrar_resultados(y_test, y_predRL)

#Gausiano
print("Métricas gausiano: ")
mostrar_resultados(y_test, y_predGA)

#Perceptrón Multicapa
print("Métricas perceptrón multicapa: ")
mostrar_resultados(y_test, y_predPM)

#Vecinos Cercanos
print("Métricas vecinos cercanos: ")
mostrar_resultados(y_test, y_predVC)

#Árbol de Decisión
print("Métricas árbol de decisión: ")
mostrar_resultados(y_test, y_predAD)


# 7. Mostrar las 5 curvas de ROC en el mismo gráfico
plt.figure(figsize=(25, 10))

#Regresion Logistica

plot_roc_curve(fprRL, tprRL, label='ROC regresión logistica')


#Gausiano

plot_roc_curve(fprGA, tprGA, label='ROC gausiano')

#Perceptron Multicapa

plot_roc_curve(fprPM, tprPM, label='ROC preceptrón multicapa')


#Vecinos Cercanos

plot_roc_curve(fprVC, tprVC, label='ROC vecinos cercanos')

#Árbol de Decision

plot_roc_curve(fprAD, tprAD, label='ROC árbol de decisión')


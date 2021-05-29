# -*- coding: utf-8 -*-
"""
Created on Sat May 29 14:35:35 2021

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

#Reemplazar los valores que son NaN
data.Glucose.replace(0, 121, inplace=True)

data.BloodPressure.replace(0, 69, inplace=True)

data.SkinThickness.replace(0, 21, inplace=True)

data.Insulin.replace(0, 80, inplace=True)

data.BMI.replace(0, 31, inplace=True)

#confirmamos que todo este correcto
data.info()

##definimos los rangos para agrupar la data
#definimos los rangos de grosor de la piel
rangosPiel = [0, 10, 15, 20, 25, 30, 35, 40, 100]
nombresPiel = ['1', '2', '3', '4', '5', '6', '7', '8']
data.SkinThickness = pd.cut(data.SkinThickness, rangosPiel, labels=nombresPiel)

#definimos los rangos de glucosa
rangosGlucosa = [40, 80, 100, 120, 140, 160, 200]
nombresGlucosa = ['1', '2', '3', '4', '5', '6']
data.Glucose = pd.cut(data.Glucose, rangosGlucosa, labels=nombresGlucosa)

#definimos los rangos de presión sanguinea
rangosPresion = [20, 50, 60, 65, 70, 75, 80, 90, 100, 125]
nombresPresion = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
data.BloodPressure = pd.cut(data.BloodPressure, rangosPresion, labels=nombresPresion)

#definimos los rangos de insulina
rangosInsulina = [10, 75, 80, 100, 150, 200, 900]
nombresInsulina = ['1', '2', '3', '4', '5', '6']
data.Insulin = pd.cut(data.Insulin, rangosInsulina, labels=nombresInsulina)

#definimos los rangos de IMC
rangosIMC = [0, 18.5, 25, 27, 35, 40, 50, 70]
nombresIMC = ['1', '2', '3', '4', '5', '6', '7']
data.BMI = pd.cut(data.BMI, rangosIMC, labels=nombresIMC)

#definimos los rangos de edad
rangosEdad = [20, 30, 40, 50, 60, 70, 80, 90]
nombresEdad = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangosEdad, labels=nombresEdad)

#definimos los rangos de dia
rangosDia = [0, 0.5, 1, 1.5, 2, 2.5]
nombresDia = ['1', '2', '3', '4', '5']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rangosDia, labels=nombresDia)

##Entrenamiento y validación cruzada
# 0 = No tiene diabetes y 1 = tiene diabetes
x = np.array(data.drop(['Outcome'], 1))
y = np.array(data.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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

def show_roc_curve_matrix(fpr, tpr, matriz_confusion):
    sns.heatmap(matriz_confusion)
    plt.show()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
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


# 1. Entrenar 5 modelos con distintos algoritmos de Machine Learning
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

#Gausiano
print("Matriz de confusión gausiano:")
print(confusion_matrix( y_test, y_predGA))

#Perceptrón Multicapa
print("Matriz de confusión perceptrón multicapa:")
print(confusion_matrix( y_test, y_predPM))

#Vecinos Cercanos
print("Matriz de confusión vecinos cercanos:")
print(confusion_matrix( y_test, y_predVC))

#Árbol de Decisión
print("Matriz de confusión árbol de decisión:")
print(confusion_matrix( y_test, y_predAD))


# 5. Imprimir las 5 matrices de confusión en un solo gráfico, empleando el mapa de calor de la librería Seaborn
fig = plt.figure(figsize=(11, 15))

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)

sns.heatmap(matriz_confusionRL, ax=ax1, cmap="YlGnBu", annot=True)
sns.heatmap(matriz_confusionGA, ax=ax2, cmap="YlGnBu", annot=True)
sns.heatmap(matriz_confusionPM, ax=ax3, cmap="YlGnBu", annot=True)
sns.heatmap(matriz_confusionVC, ax=ax4, cmap="YlGnBu", annot=True)
sns.heatmap(matriz_confusionAD, ax=ax5, cmap="YlGnBu", annot=True)
plt.show()

## 6. Imprimir las métricas de precisión, recall y f1 de cada clase de cada modelo
#Regresión Logistica
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
probRL = modelRL.predict_proba(x_test)
probRL = probRL[:, 1]
auc = roc_auc_score(y_test, probRL)
fprRL, tprRL, _ = roc_curve(y_test, probRL)
plot_roc_curve(fprRL, tprRL, label='ROC regresión logistica')


#Gausiano
probGA = modelGA.predict_proba(x_test)
probGA = probGA[:, 1]
auc = roc_auc_score(y_test, probGA)
fprGA, tprGA, _ = roc_curve(y_test, probGA)
plot_roc_curve(fprGA, tprGA, label='ROC gausiano')

#Perceptron Multicapa
probPM = modelPM.predict_proba(x_test)
probPM = probPM[:, 1]
auc = roc_auc_score(y_test, probPM)
fprPM, tprPM, _ = roc_curve(y_test, probPM)
plot_roc_curve(fprPM, tprPM, label='ROC preceptrón multicapa')


#Vecinos Cercanos
probVC = modelVC.predict_proba(x_test)
probVC = probVC[:, 1]
auc = roc_auc_score(y_test, probVC)
fprVC, tprVC, _ = roc_curve(y_test, probVC)
plot_roc_curve(fprVC, tprVC, label='ROC vecinos cercanos')

#Árbol de Decision
probAD = modelAD.predict_proba(x_test)
probAD = probAD[:, 1]
auc = roc_auc_score(y_test, probAD)
fprAD, tprAD, _ = roc_curve(y_test, probAD)
plot_roc_curve(fprAD, tprAD, label='ROC árbol de decisión')


plt.plot([0, 1], [0, 1], color='black', label='Línea de NO discriminacion')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curvas de ROC')
plt.legend()
plt.show()

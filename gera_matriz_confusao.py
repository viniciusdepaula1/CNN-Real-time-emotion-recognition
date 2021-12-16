import json
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.saving.model_config import model_from_json
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#aula29 tem acuracia e erro

#gerando matriz de confusão

#conjunto dos pesos que a rede neural aprendeu
arquivo_modelo = 'resources\\modelo_02_expressoes.h5'

#estrutura da rede neural
arquivo_modelo_json = 'resources\\modelo_02_expressoes.json'

true_y = []  #respostas reais
pred_y = []  #previsões

#conjunto de testes e repostas reais
x = np.load('resources\\mod_xtest2.npy')     
y = np.load('resources\\mod_ytest2.npy')     

#print(x[0], y[0])  #imagem 1 e resposta 1


#Carregamento do modelo

json_file = open(arquivo_modelo_json, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json);
loaded_model.load_weights(arquivo_modelo);

#Gerando previsões da rede
y_pred = loaded_model.predict(x);

yp = y_pred.tolist() #previsões
yt = y.tolist()      #respostas reais na base de testes
count = 0

#calculo manual do accuracy
for i in range(len(y)):
    yy = max(yp[i])     #maior valor de probabilidade da rede
    yyt = max(yt[i])    #maior valor nas respostas reais

    pred_y.append(yp[i].index(yy))  #associação
    true_y.append(yt[i].index(yyt))

    if(yp[i].index(yy) == yt[i].index(yyt)):
        count += 1

acc = (count / len(y)) * 100

print('Acurácia no conjunto de teste: ', str(acc))

cm = confusion_matrix(true_y, pred_y)
expressoes = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro'];
titulo = 'Matriz de confusão'

cm = cm / cm.astype(np.float).sum(axis=1)

cmn = cm.astype('float') 
cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=expressoes, yticklabels=expressoes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
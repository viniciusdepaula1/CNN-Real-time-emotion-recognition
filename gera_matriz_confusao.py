import json
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.saving.model_config import model_from_json
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
#aula29 tem acuracia e erro

#gerando matriz de confusão

#conjunto dos pesos que a rede neural aprendeu
arquivo_modelo = 'resources\\modelo_06_expressoesVGG.h5'

#estrutura da rede neural
arquivo_modelo_json = 'resources\\modelo_06_expressoesVGG.json'

true_y = []  #respostas reais
pred_y = []  #previsões

x = np.load('resources\\mod_xtest.npy')
y = np.load('resources\\mod_ytest.npy')

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

#np.save('novo_truey_mod02', true_y)  #valores reais
#np.save('novo_predy_mod02', pred_y)  #valores previstos

cm = confusion_matrix(true_y, pred_y)
expressoes = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro'];
titulo = 'Matriz de confusão'
print(cm)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
plt.title(titulo)
plt.colorbar()

tick_marks = np.arange(len(expressoes))

plt.xticks(tick_marks, expressoes, rotation = 45)
plt.yticks(tick_marks, expressoes)

fmt = 'd'
thresh = cm.max()/2.

for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
        color='white' if cm[i, j] > thresh else 'black')

plt.ylabel('Classificação correta')
plt.xlabel('Predição')
plt.savefig('matriz_confusao_mod03.pdf')

plt.show()
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#captura da webcan ou video
vid = cv.VideoCapture('resources\\video_teste06.mp4');
#vid = cv.VideoCapture(0);

#arquivo específico para trabalhar com detecção de faces
cascade_faces = "resources\\haarcascade_frontalface_default.xml";

#modelo da rede neural já pré treinada
caminho_modelo = "resources\\modelo_06_expressoesVGG.h5";

#usando o classificador para fazer a detecção de faces
face_detection = cv.CascadeClassifier(cascade_faces);

#carregando um modelo salvo do TensorFlow
classificador_emocoes = load_model(caminho_modelo, compile= False);

while(True):
  ret, frame = vid.read()

  if(ret == False):
    break;

  #emocoes em formato de lista para classificacao
  expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", 
      "Surpreso", "Neutro"];  

  #Converter para escala de cinza, mais rapido para a rede neural
  cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  #armazenando todas as faces detectadas na imagem
  faces = face_detection.detectMultiScale(frame, scaleFactor = 1.2,
    minNeighbors = 5, minSize = (20, 20))

  if len(faces) > 0:
    for (x, y, w, h) in faces:
      
      regiao_da_face = cinza[y:y + h, x:x + w] #coordenadas da face  
      regiao_da_face = cv.resize(regiao_da_face, (48, 48)) #redimensiona imagem

      #normalizacao entre 0 e 1
      regiao_da_face = regiao_da_face.astype('float')
      regiao_da_face = regiao_da_face / 255
      regiao_da_face = img_to_array(regiao_da_face)
      regiao_da_face = np.expand_dims(regiao_da_face, axis = 0) #adiciona mais uma dimensão (1, 48, 48, 1)

      #previsoes
      preds = classificador_emocoes.predict(regiao_da_face)[0]
      emotion_prob = np.max(preds)
      label = expressoes[preds.argmax()]

      #escrevendo no frame o resultado
      cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
          0.65, (0, 0, 255), 2, cv.LINE_AA)

      #escrevendo o retangulo na face
      cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
  #else:
    #print('Nenhuma face encontrada')

  cv.imshow('webcan', frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

vid.release()
cv.destroyAllWindows()
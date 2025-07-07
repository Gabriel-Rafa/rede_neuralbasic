#Primeira rede neural com 3 neurônios
import keras
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


df = pd.read_csv('/home/rafalinux/Downloads/estudopython/dataset/admission_dataset.csv')

y = df['Chance of Admit ']
x = df.drop('Chance of Admit ', axis=1)

x_train, x_test = x[0:300], x[300:]
y_train, y_test = y[0:300], y[300:]

print( x_test.shape)



#criando a arquitetura da rede neural
modelo = Sequential()
modelo.add(Dense(units=3, activation='relu', input_dim=x_train.shape[1]))
modelo.add(Dense(units=1, activation='linear'))

#treinando a rede neural
modelo.compile(loss='mse', optimizer='adam', metrics=['mae'])
resultado = modelo.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test))

 #plotar gráfico do historico de treinamento
plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Historico de Treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Época de Treinamento')
plt.legend(['Erro de treino', 'Erro de teste'])
plt.show()
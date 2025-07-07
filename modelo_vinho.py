import pandas as pd

arquivo = pd.read_csv('/home/rafalinux/Downloads/estudopython/dataset/wine_dataset.csv')

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

y = arquivo['style']
x = arquivo.drop('style', axis=1)

from sklearn.model_selection import train_test_split
#criando os conjuntos de dados de treino e teste:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier
#criação do modelo
modelo = ExtraTreesClassifier()
modelo.fit(x_train, y_train)

#imprimindo os resultados:
resultado = modelo.score(x_test, y_test)
#print("Acurácia:", resultado)

x_test[400:403]
print(y_test[400:403])
previsoes = modelo.predict(x_test[400:403])

print(previsoes)
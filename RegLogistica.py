import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#%%

# Carregando a base de dados, visualização de pontos e visualização estatística
df = pd.read_csv('Eleicao.csv', sep = ';')
plt.scatter(df.DESPESAS, df.SITUACAO)
desc = df.describe()

#%%

# Visualização do coeficiente de correlação entre o atributo "despesas" e "situação"
coef = np.corrcoef(df.DESPESAS, df.SITUACAO)

#%%

# Criação das variáveis x e y (variável independente e variável dependente)
# Transformação de x para o formato de matriz adicionando um novo eixo (newaxis)
x = df.iloc[:, 2].values
x = x[:, np.newaxis]
y = df.iloc[:, 1].values

#%%

# Criação do modelo, treinamento e visualização dos coeficientes
modelo = LogisticRegression()
modelo.fit(x, y)
m_coef = modelo.coef_
m_interc = modelo.intercept_

#%%

# Geração de novos dados para gerar a função sigmoide
x_teste = np.linspace(10, 3000, 100)
# Implementação da função sigmoide
def model(x):
    return 1 / (1+np.exp(-x))
# Geração de previsões (variável r) e visualização dos resultados
r = model(x_teste * m_coef + m_interc).ravel()
plt.scatter(df.DESPESAS, df.SITUACAO)
plt.plot(x_teste, r, color = 'red')

#%%

# Carregamento da base de dados com os novos candidatos
df_prev = pd.read_csv('NovosCandidatos.csv', sep = ';')

#%%

# Mudança dos dados para formato de matriz
despesas = df_prev.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)
# Previsões e geração de nova base de dados com os valores originais e as previsões
prev_teste = modelo.predict(despesas)

#%%

df_prev = np.column_stack((df_prev, prev_teste))


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# função para treinar o modelo de regressão logística
def treinar_modelo(dados, coluna_texto, coluna_sentimento):
  # faz o bag-of-words para separadar as palavras mais frequentes do texto
  vetorizar = CountVectorizer(max_features=100)
  bag_of_words = vetorizar.fit_transform(dados[coluna_texto])

 # divide o dataset em treino e teste
  X_train, X_test, y_train, y_test = train_test_split(bag_of_words,
                                                    dados[coluna_sentimento],
                                                    stratify=dados[coluna_sentimento],
                                                    random_state = 71)

  # treina o modelo de regressão logística
  regressao_logistica = LogisticRegression()
  regressao_logistica.fit(X_train, y_train)

  # retorna a acurácia do modelo
  return regressao_logistica.score(X_test, y_test)

# função para retornar as palavras das avaliações negativas
def obter_palavras_negativas(dados, coluna_texto):
  texto_negativo = dados.query("polarity == 0")
  todas_avaliacoes = list(texto_negativo[coluna_texto])
  return ' '.join(todas_avaliacoes)

# função para retornar as palavras das avaliações positivas
def obter_palavras_positivas(dados, coluna_texto):
  texto_positivo = dados.query("polarity == 1")
  todas_avaliacoes = list(texto_positivo[coluna_texto])
  return ' '.join(todas_avaliacoes)

# função para retornar as palavras de todas as avaliações (positivas e negativas)
def obter_todas_palavras(dados, coluna_texto):
  todas_avaliacoes = list(dados[coluna_texto])
  return ' '.join(todas_avaliacoes)

# função para gerar a nuvem de palavras a partir de um texto
def gerar_nuvem_palavras(texto):
  return WordCloud(width=800, height=500,
    max_font_size=110,
    collocations=False).generate(texto)

# função para exibir a nuvem de palavras
def exibir_nuvem_palavras(nuvem_palavras, titulo="Nuvem de palavras"):
  plt.figure(figsize=(10,7), num=titulo)
  plt.imshow(nuvem_palavras, interpolation='bilinear')
  plt.title(titulo)
  plt.axis("off")

avaliacoes = pd.read_csv("data/Brazilian-Portuguese-Sentiment-Analysis-Datasets.csv")
print('\n\n')
print("Arquivo CSV carregado com sucesso!")
print(avaliacoes.head())
print('\n\n')

# remover colunas desnecessárias
avaliacoes = avaliacoes.drop(["original_index", "review_text_processed", "review_text_tokenized", "rating", "kfold_polarity", "kfold_rating"], axis=1)
print("Colunas desnecessárias removidas com sucesso!")
print(avaliacoes.head())
print('\n\n')

# remover linhas com valores nulos
avaliacoes.dropna(inplace=True, axis=0)
print("Linhas com valores nulos removidas com sucesso!")
print(avaliacoes.head())
print('\n\n')

# contar a quantidade de avaliações positivas, negativas e neutras
avaliacoes.polarity.value_counts()
print("Contagem de avaliações por polaridade:")
print(avaliacoes.polarity.value_counts())

# treinar o modelo e calcular a acurácia
acuracia = treinar_modelo(avaliacoes, "review_text", "polarity")
print(f"Acurácia do modelo: {acuracia}")

# gerar a lista de palavras negativas, positivas e totais
lista_palavras_negativas = obter_palavras_negativas(avaliacoes, "review_text")
lista_palavras_totais = obter_todas_palavras(avaliacoes, "review_text")
lista_palavras_positivas = obter_palavras_positivas(avaliacoes, "review_text")

# gerar a 'url' da nuvem de palavras a partir das listas de palavras
nuvem_palavras_negativas = gerar_nuvem_palavras(lista_palavras_negativas)
nuvem_palavras_totais = gerar_nuvem_palavras(lista_palavras_totais)
nuvem_palavras_positivas = gerar_nuvem_palavras(lista_palavras_positivas)

# exibir as nuvens de palavras (todas juntas, em janelas separadas)
exibir_nuvem_palavras(nuvem_palavras_negativas, "Avaliações negativas")
exibir_nuvem_palavras(nuvem_palavras_totais, "Todas as avaliações")
exibir_nuvem_palavras(nuvem_palavras_positivas, "Avaliações positivas")
plt.show()

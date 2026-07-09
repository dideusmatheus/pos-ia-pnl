
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

# gerar a nuvem de palavras (word cloud) com todas as palavras das avaliações
todas_avaliacoes = list(avaliacoes.review_text)
todas_palavras = ' '.join(todas_avaliacoes)
print("Todas as palavras das avaliações foram unidas com sucesso!")
print(len(todas_palavras))

# gerar a nuvem de palavras
nuvem_palavras = WordCloud(width=800, height=500,
  max_font_size=110,
  collocations=False).generate(todas_palavras)
print("Nuvem de palavras gerada com sucesso!")
print(nuvem_palavras)

# exibir a nuvem de palavras
plt.figure(figsize=(10,7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis("off")
plt.show()





























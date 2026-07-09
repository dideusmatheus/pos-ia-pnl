
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

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

# dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(avaliacoes["review_text"], avaliacoes["polarity"], stratify=avaliacoes.polarity, random_state=71)

# regressao_logistica = LogisticRegression()
# regressao_logistica.fit(treino, classe_treino)
# acuracia = regressao_logistica.score(teste, classe_teste)
# print(acuracia)


# texto = ["Este produto é muito bom", "Este produto é muito ruim"]
# vetorizar = CountVectorizer()
# bag_of_words = vetorizar.fit_transform(texto)
# print(bag_of_words)

# matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words,
#   columns=vetorizar.get_feature_names_out())
# print(matriz_esparsa)

# 
vetorizar = CountVectorizer(max_features=100)
bag_of_words = vetorizar.fit_transform(avaliacoes.review_text)
print(bag_of_words.shape)













































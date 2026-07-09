
import pandas as pd

# carregar o arquivo CSV com as avaliações
avaliacoes = pd.read_csv("data/Brazilian-Portuguese-Sentiment-Analysis-Datasets.csv")
print(avaliacoes.head())

# remover colunas desnecessárias
avaliacoes = avaliacoes.drop(["original_index", "review_text_processed", "review_text_tokenized", "rating", "kfold_polarity", "kfold_rating"], axis=1)
print(avaliacoes.head())

# remover linhas com valores nulos
avaliacoes.dropna(inplace=True, axis=0)
print(avaliacoes.head())



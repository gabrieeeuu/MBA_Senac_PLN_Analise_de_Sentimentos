import streamlit as st
from googletrans import Translator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re

try:
  stopwords.words('portuguese')
except LookupError:
  nltk.download('stopwords')
try:
  word_tokenize("exemplo")
except LookupError:
  nltk.download('punkt_tab')

# lib WordCloud para nuvem de palavras
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Inicializações
translator = Translator()
sia = SentimentIntensityAnalyzer()
stop_words_pt = set(stopwords.words('portuguese'))

bag_of_words = [
    "desperdício", "comida", "alimento", "doar", "doação", "restaurante",
    "hotel", "buffet", "sobra", "excesso", "consciência", "reaproveitamento"
]

lexico_sentimentos = {
    "eficiência": 2.0,
    "controle": 1.8,
    "solidariedade": 2.5,
    "reaproveitamento": 2.2,
    "conscientização": 2.0,
    "educar": 1.5,
    "impacto": -0.8,
    "ambiental": 0.5,
    "treinamentos": 1.0,
    "tecnologia": 1.3,
    "preventiva": 1.7,
    "insatisfação": -2.0,
    "consultar": 0.3,
    "evitar": -0.5,
    "validade": 0.4,
    "descartar": -1.5,
    "sobras": -1.0,
    "higiene": 1.0,
    "descontos": 1.2,
    "engajamento": 1.8,
    "consumidor": 1.0,
    "modelo": 1.0,
    "preditivo": 1.3,
    "sazonalidade": 0.6,
    "inovação": 2.0,
    "prevenção": 1.5,
    "ignorar": -2.5,
    "apresentação": 0.5,
    "problema": -1.3,
    "desperdício": -2.0,
    "alimentar": 0.4
}

# Funções
def preprocessar_texto(texto):
  texto = texto.lower()
  texto = re.sub(r"[^\w\s]", "", texto)
  texto = re.sub(r'http\S+|@\S+|#\S+', '', texto) # Remover URLs, menções e hashtags
  tokens = word_tokenize(texto)
  tokens = [w for w in tokens if w.isalpha() and w not in stop_words_pt]
  return tokens

def comentario_relevante(texto, bow):
  texto = preprocessar_texto(texto)
  return any(palavra in texto for palavra in bow)

def analisar_sentimento(texto_processado, lexico):
  palavras = texto_processado.split()
  score = sum(lexico.get(p, 0) for p in palavras)
  if score > 0:
    return "Positivo", score
  elif score < 0:
    return "Negativo", score
  else:
    return "Neutro", score

# Título e descrição do projeto
st.title("Análise de Sentimento de Ideias de Projetos")
st.write("Este app traduz sua ideia para o inglês e analisa o sentimento com VADER.")

# Entradas do usuário
uploaded_file = st.file_uploader("Faça upload de um CSV com comentários (coluna 'comentario')", type="csv")

texto = st.text_area("Digite a ideia do projeto:", height=100)

if st.button("Analisar Sentimento"):
  if not texto.strip():
    st.warning("Por favor, insira um texto para análise.")
  elif uploaded_file is None:
    st.warning("Por favor, envie um arquivo CSV com comentários.")
  else:
    df = pd.read_csv(uploaded_file)
    comentarios = df['comentario'].astype(str).tolist()
    comentarios_relevantes = [c for c in comentarios if comentario_relevante(c, bag_of_words)]

    # --- VADER ---
    traducao = translator.translate(texto, src='pt', dest='en').text
    scores_vader = sia.polarity_scores(traducao)
    comp = scores_vader['compound']
    resultado_vader = "Positivo" if comp >= 0.05 else "Negativo" if comp <= -0.05 else "Neutro"

    # --- Personalizado ---
    sentimento_personalizado, score_personalizado = analisar_sentimento(texto, lexico_sentimentos)

    # --- Resultados ---
    st.markdown("## 🔎 Resultados da Análise")
    st.markdown("**VADER (Inglês)**: " + resultado_vader)
    st.markdown("**Análise com Léxico Personalizado**: " + sentimento_personalizado + f" (pontuação: {score_personalizado})")
    st.json(scores_vader)

    # --- Nuvem de Palavras ---
    st.markdown("## ☁️ Nuvem de Palavras dos Comentários")
    todos_textos = [preprocessar_texto(c) for c in comentarios_relevantes]
    todos_textos = ' '.join([' '.join(texto) for texto in todos_textos])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(todos_textos)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

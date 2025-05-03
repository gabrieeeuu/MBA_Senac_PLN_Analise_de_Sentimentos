# lib streamlit para visualização
import streamlit as st

# lib Translator para tradução
from googletrans import Translator

# lib nltk para análise de sentimento
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# lib WordCloud para nuvem de palavras
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Inicializações
translator = Translator()
sia = SentimentIntensityAnalyzer()

st.title("Análise de Sentimento de Ideias de Projetos")
st.write("Este app traduz sua ideia para o inglês e analisa o sentimento com VADER.")

# Entrada do usuário
texto = st.text_area("Digite a descrição da ideia de projeto (em português):", height=100)

if st.button("Analisar Sentimento"):
    if not texto.strip():
        st.warning("Por favor, insira um texto para análise.")
    else:
        # Tradução do português para inglês
        traducao = translator.translate(texto, src='pt', dest='en')
        texto_ingles = traducao.text
        st.markdown("**Texto traduzido (inglês):**")
        st.write(texto_ingles)

        # Análise de sentimento com VADER
        scores = sia.polarity_scores(texto_ingles)
        compound = scores['compound']

        if compound >= 0.05:
            sentimento = "Positivo"
        elif compound <= -0.05:
            sentimento = "Negativo"
        else:
            sentimento = "Neutro"

        st.markdown("### Resultado da Análise")
        st.write(f"##### Sentimento classificado: **{sentimento}**")
        st.json(scores)
        
        # Palavras da nuvem em inglês
        words_en = texto_ingles.lower().split()
        unique_words = list(set(words_en))

        # Nuvem de palavras
        st.markdown("### ☁️ Nuvem de Palavras em Português (baseada na ideia)")
        
        # Tradução das palavras para português
        translated_words = []
        for word in unique_words:
            try:
                translated = translator.translate(word, src='en', dest='pt').text
                translated_words.append(translated)
            except:
                continue  # ignora erros de tradução

        # Geração da nuvem com palavras traduzidas
        text_pt = ' '.join(translated_words)
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', colormap='plasma'
        ).generate(text_pt)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
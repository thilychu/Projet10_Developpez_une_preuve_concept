import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import requests
from wordcloud import WordCloud,STOPWORDS


    
@st.cache_data()
def load_data():
    df = pd.read_csv('data_db.csv')
    return df

def styled_dataframe_description(dataframe, size):
    # Style the DataFrame description using HTML
    styled_html = f"<div style='font-size:{size}px;'>{dataframe.describe().to_html()}</div>"
    return st.markdown(styled_html, unsafe_allow_html=True)

def styled_text(text, size):
    return f"<span style='font-size:{size}px;'>{text}</span>"

# Get user input for text size
text_size = st.sidebar.slider("Ajuster la taille du texte", min_value=10, max_value=30, step=1, value=20)
text_table = st.sidebar.slider("Adjuster la taille su texte dans tableau", min_value=10, max_value=30, step=1, value=20)

            
# Function for EDA of tweet data
def eda(data):
    st.header("Analyse exploratoire des données des tweets")
    st.markdown(styled_text("Le dataset utilisé dans ce travail est le jeu de données Sentiment140. Il contient 1 600 000 tweets extraits à l'aide de l'API Twitter. Les tweets ont été annotés (0 = négatif, 4 = positif) et peuvent être utilisés pour détecter le sentiment. Nous avons nettoyé les textes avant de les utiliser pour entrainer notre modèle. Cela impliquait de supprimer des éléments indésirables tels que les caractères spéciaux, la ponctuation excessive et les balises HTML, etc. .", text_size), unsafe_allow_html=True)
    
    st.markdown(styled_text("**Voici un aperçu des textes bruts et des textes nettoyés:**", text_size), unsafe_allow_html=True)
    styled_html = f"<div style='font-size:{text_table}px;'>{data.head().to_html()}</div>"
    st.markdown(styled_html, unsafe_allow_html=True)
    st.write("\n\n")
 
    st.markdown(styled_text("**Statistiques descriptives sur les données brutes :**", text_size), unsafe_allow_html=True)
    data['number_character'] = data['text'].apply(len)
    data['number_of_words'] = data['text'].apply(lambda x: len(x.split()))
    data['number_sentence'] = data['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    styled_dataframe_description(data, text_table)
    st.write("\n\n")

    data.drop('clean_text', axis=1, inplace=True)           
    df_neg = data[data.target=="NEGATIVE"].copy()
    df_pos = data[data.target=="POSITIVE"].copy()
    st.markdown(styled_text("**Statistiques descriptives sur les données positives :**", text_size), unsafe_allow_html=True)
    styled_dataframe_description(df_pos, text_table)
    st.write("\n\n")
    
    st.markdown(styled_text("**Statistiques descriptives sur les données négatives :**", text_size), unsafe_allow_html=True)
    styled_dataframe_description(df_neg, text_table)
    st.write("\n\n")
            
    st.markdown(styled_text("**Le graphique de distribution de la variable cible montre que le pourcentage de tweets négatifs et positifs est équivalent.**", text_size), unsafe_allow_html=True)
    # Specify custom colors for Negative and Positive bars
    colors = {'Negative': 'red', 'Positive': 'green'}
    df1 = data.groupby(['target']).count()['text']
    df1 = df1.apply(lambda x: round(x*100/len(data),3))
    df = df1.reset_index(name='% of instances').rename(columns={'index': 'target'})
    fig = px.bar(df, x='target', y='% of instances', barmode='group', color='target', color_discrete_sequence=['red', 'green'])

    # Update layout for better hover information
    fig.update_layout(
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Courier New", font_color="black"),
        showlegend=False  # No need for legend in this case
    )
    # Display the bar chart using Streamlit
    st.plotly_chart(fig)
    
    st.markdown(styled_text("La distribution des textes négatifs et positifs est presque équivalente en termes de nombre de mots et de phrases. Les tweets négatifs sont légèrement plus longs (mais pas de manière significative) que les tweets positifs, comme le montre le bar chart suivant", text_size), unsafe_allow_html=True)
    
    df = {'Category': ['Chars', 'Words', 'Sentences'],
            'Negative': [df_neg['number_character'].mean(),df_neg['number_of_words'].mean(), df_neg['number_sentence'].mean()],
            'Positive': [df_pos['number_character'].mean(),df_pos['number_of_words'].mean(),df_pos['number_sentence'].mean()]}
    df = pd.DataFrame(df)
   
    # Create a Plotly bar chart
    fig = px.bar(df, x='Category', y=['Negative', 'Positive'], barmode='group', color_discrete_map=colors)

    # Update layout for better hover information
    fig.update_layout(
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Courier New",font_color="black"),
        showlegend=True
    )
    st.plotly_chart(fig)
    
    plt.figure(figsize=(15, 15))
    st.markdown(styled_text("Ce wordcloud représente le nuage de mots pour tous les textes :", text_size), unsafe_allow_html=True)
    # Generate the word cloud for all data
    all_text=" ".join(data['text'].values.tolist())
    
    wordcloud = WordCloud(width=800,height=800,stopwords=STOPWORDS,background_color='black',
                          max_words=800,colormap="Spectral").generate(all_text)
    st.image(wordcloud.to_array())
    
    st.markdown(styled_text("Ce wordcloud représente le nuage de mots pour tous les tweets négatives :", text_size), unsafe_allow_html=True)
    # Generate the word cloud for negatives
    neg_wordcloud=data[data["target"]=='NEGATIVE']
    neg_text=" ".join(neg_wordcloud['text'].values.tolist())
    wordcloud = WordCloud(width=800, height=800,stopwords=STOPWORDS, background_color='black',max_words=800,colormap="Spectral").generate(neg_text)              
    st.image(wordcloud.to_array())
    
    st.markdown(styled_text("Ce wordcloud représente le nuage de mots pour tous les tweets positives :", text_size), unsafe_allow_html=True)

      # Generate the word cloud for positives
    pos_wordcloud=data[data["target"]=='POSITIVE']
    pos_text=" ".join(pos_wordcloud['text'].values.tolist())
    wordcloud = WordCloud(width=800, height=800,stopwords=STOPWORDS, background_color='black',max_words=800,colormap="Spectral").generate(pos_text)              
    st.image(wordcloud.to_array())  
          
# Function for sentiment analysis prediction
def predict_sentiment(url):
    st.header("Analyse de Sentiment")
	
    # Sample tweet input for prediction
    text = st.text_area("Enter a tweet for sentiment analysis:")
    
    if st.button("Predict Sentiment"):
    	try:
            response = requests.get(url)
            response = requests.post(url, data={"name": text})
            result = response.json()
            st.write("Prediction Result:")
            st.write(result['result'])
    	except requests.exceptions.RequestException as e:
            st.error(f"Error during URL call: {e}")

# Main function
def main():
    st.title("Tableau de bord d'Analyse de Sentiment")

    # Load tweet data
    tweet_data = load_data()
    url = "https://4b6e-213-55-220-85.ngrok-free.app//predict"

    # Sidebar navigation
    selected_page = st.sidebar.selectbox("Choisir une page", options=["EDA", "Analyse de Sentiment"])

    # Page content
    if selected_page == "EDA":
        eda(tweet_data)
    elif selected_page == "Analyse de Sentiment":
        predict_sentiment(url)

if __name__ == "__main__":
    main()

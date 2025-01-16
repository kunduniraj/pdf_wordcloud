import streamlit as st
import PyPDF2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# Download NLTK data if not already present
nltk.download("punkt")
nltk.download("stopwords")

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocesses the text by converting to lowercase, removing special characters, and tokenizing."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = word_tokenize(text)
    return tokens

def remove_stop_words(tokens):
    """Removes stop words from the tokenized text."""
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def create_wordcloud(tokens):
    """Generates a word cloud from the given tokens."""
    wordcloud = WordCloud(width=800, height=400, max_words=500, background_color="white").generate(" ".join(tokens))
    return wordcloud

def main():
    st.title("PDF Text WordCloud Generator")

    st.sidebar.header("Upload PDF File")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        st.subheader("Extracted Text")
        st.text_area("Text", text[:2000], height=200)  # Display first 2000 characters

        with st.spinner("Processing text..."):
            tokens = preprocess_text(text)
            filtered_tokens = remove_stop_words(tokens)

        st.subheader("Word Cloud")
        wordcloud = create_wordcloud(filtered_tokens)

        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

if __name__ == "__main__":
    main()

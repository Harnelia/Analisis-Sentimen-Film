from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Extended positive and negative word lists
positive_words = [
    "pas", "bagus", "cocok", "mirip", "pass", "asik", "rezeki", "biasa","sabar",
    "positif", "luar biasa", "semirip", "definisi", "peran asli", "paling bener", 
    "paling cocok", "melebihi ekspektasi", "booming", "secakep", "related", 
    "ternyata nyatanya", "sesuai ekspektasi", "pas banget", "adil", "sampe puas",
    "keren","suka","seru","kaget","anti mainstream", "rekomendasi","terima kasih",
    "penasaran","smooth","bagus","coba","senang","banget","asli","best","lagi","bagusan",
    "banget","tonton","kocak","gas","ayo","mantap","kocak","keren","sudah","menarik","sudah","rezeki",
    "ketawa","seseru","wah","lucu"
]

negative_words = [
    "kecewa", "sakit", "menyesal", "ngambek", "nangis", "nyawa", "tidak","malah","putih pucat","dilan","mantannya",
    "jauh", "meleset", "kecewa berat", "ntah", "sia-sia", "gagal", "tidak rela","kalau tidak","laura, maudy","tan skin","ketuaan",
    "sulit", "susah", "tidak paham", "anjir", "protes", "benci", "kocak", "bete","cuman","caitlin","bikin",
    "mentalnya", "malesan", "redup", "gelap", "gosong", "kehilangan", "tan skin","dania","putih abu","jauh banget","heran","sok","ekspektasi",
    "magrib", "ati", "gila", "nangis","cape", "kecewa ","ekspetasi","aisyah","mudah saja","bukan",
    "tidak sanggup", "putih abu-abu","mata pencaharian", "mata teduh", "mata batin", "gamon", "putih pucat bening", 
    "kulit", "berat", "krna","tidak expect","kurang", "tidak ada","kurang paham", 
    "bingung","tidak ada","jumpscare","tidak ada","seram","tidak nyambung","hilang","milih","ganti","terus",
    
]

def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.lower()
    return text

def normalize_text(text):
    return text

def tokenize_text(text):
    return text.split()

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def filter_text(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [token for token in tokens if token not in stop_words]

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    tokens = lemmatize_text(tokens)
    tokens = filter_text(tokens)
    return ' '.join(tokens)

def compute_sentiment_score(text):
    vectorizer = TfidfVectorizer()
    
    try:
        vectors = vectorizer.fit_transform([text])
    except ValueError:
        return "Unknown"  # Return a default sentiment if vectorizer fails
    
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    
    positive_score = sum(df[word].sum() for word in positive_words if word in df)
    negative_score = sum(df[word].sum() for word in negative_words if word in df)
    
    if positive_score > negative_score:
        return "Positive"
    elif positive_score < negative_score:
        return "Negative"
    else:
        return "Netral"  # Return 'Neutral' if scores are equal

@app.route('/')
def index():
    return render_template('index.html', movie_name='')

@app.route('/predict', methods=['POST'])
def predict():
    movie_name = request.form['movie_name']
    
    data = pd.read_csv('reviews.csv', encoding='ISO-8859-1')
    
    movie_reviews = data[data['movie_name'].str.lower() == movie_name.lower()]
    
    if movie_reviews.empty:
        return render_template('index.html', movie_name=movie_name, reviews=[], message="No reviews found for this movie.")
    
    movie_reviews['processed_comment'] = movie_reviews['comment'].apply(preprocess_text)
    movie_reviews['sentiment'] = movie_reviews['processed_comment'].apply(compute_sentiment_score)
    
    # Tambahkan kolom nomor
    movie_reviews['number'] = range(1, len(movie_reviews) + 1)
    
    reviews_list = movie_reviews[['number', 'user_name', 'comment', 'sentiment']].to_dict('records')
    
    # Hitung jumlah komentar positif dan negatif
    positive_count = movie_reviews[movie_reviews['sentiment'] == 'Positive'].shape[0]
    negative_count = movie_reviews[movie_reviews['sentiment'] == 'Negative'].shape[0]
    netral_count = movie_reviews[movie_reviews['sentiment'] == 'Netral'].shape[0]
    
    # Berikan rekomendasi berdasarkan jumlah komentar positif dan negatif
    if positive_count > negative_count and positive_count > netral_count:
        recommendation = "Film ini sangat direkomendasikan!"
    elif negative_count > positive_count and negative_count > netral_count:
        recommendation = "Film ini kurang direkomendasikan." 
    elif netral_count > positive_count and netral_count > negative_count:
        recommendation = "Film ini direkomendasikan"
    else:
        recommendation = "Film ini memiliki sentimen campuran."
    
    return render_template('index.html', movie_name=movie_name, reviews=reviews_list, positive_count=positive_count, negative_count=negative_count, netral_count=netral_count,recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)

import re
import string
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
import spacy
# from textblob import TextBlob
from bs4 import BeautifulSoup 
import emoji
import os
from multiprocessing import Pool,cpu_count
# from spellchecker import SpellChecker
import json
import pandas as pd
import openpyxl
import csv

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
# lemmatizer = WordNetLemmatizer()

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

curr_dir = os.path.dirname(__file__)
data_path = os.path.join(os.path.dirname(os.path.dirname(curr_dir)), 'data')
output_file_path= os.path.join(data_path, 'preprocessed_data','preprocessed_data_facebook.txt')
input_file_path = os.path.join(data_path,'facebookdata.xlsx')
df = pd.read_excel(input_file_path,engine='openpyxl')

with open('slang.json', 'r') as json_file:
        json_string = json_file.read()
json_data = json.loads(json_string)

def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Convert emojis to text
    text = emoji.demojize(text)

    #remove html tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    #remove contractions
    text = contractions.fix(text)

    pattern_http = r'https://\S+'
    text = re.sub(pattern_http,'',text)
    #removal of punctuations
    special_chars = re.escape(string.punctuation)
    text = re.sub(r'\\u[\da-fA-F]{4}', '', text) # get all the possible list of special chars
    text = re.sub(r'['+special_chars+']','',text) # replaing special chars
    text = re.sub(r'[^\w\s]','',text) # replace the punctuations
    text = re.sub(r'\n\s*', ' ', text) # to replace extra space and empty lines
    text = text.replace('rt', '')

    #removal of numbers
    text = re.sub(r'[\d]','',text)

    # Remove non-ASCII characters from the text
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Spelling correction using TextBlob
    # correction_cache = {}
    # spell = SpellChecker()
    # def correct_text(text):
    #     # Check if the corrected result is already cached
    #     if text in correction_cache:
    #         return correction_cache[text]
        
    #     # Split the text into words
    #     words = text.split()
        
    #     # Correct each word individually
    #     corrected_words = [str(spell.correction(word)) for word in words]
        
    #     # Join the corrected words back into a string
    #     corrected_text = ' '.join(corrected_words)
        
    #     # Cache the corrected result
    #     correction_cache[text] = corrected_text
        
    #     return corrected_text
    # text = str(TextBlob(text).correct())
    # text = correct_text(text)

    
    
    # Tokenization
    tokens = word_tokenize(text)

    # Stop words removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    #converting slang words.
    # with open('slang.json', 'r') as slang_json:
    #     slang_data=json.load(slang_json)
    tokens = [json_data[token] if token in json_data else token for token in tokens]
    
    # Lemmatization using spaCy
    lemmatized_tokens = []
    # for word in tokens:
    #     lemmatized_tokens.append(lemmatizer.lemmatize(word,pos='v'))
    doc = nlp(text)
    for token in doc:
        lemmatized_tokens.append(token.lemma_)

    return ' '.join(lemmatized_tokens)


# print(df.head())
df['Comments Text'] = df['Comments Text'].apply(preprocess_text)
df.to_csv(output_file_path,index=False,header=False)
"""This is to preprocess twitter data"""
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

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
# lemmatizer = WordNetLemmatizer()

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

curr_dir = os.path.dirname(__file__)
data_path = os.path.join(os.path.dirname(os.path.dirname(curr_dir)), 'data')

with open('slang.json', 'r') as json_file:
        json_string = json_file.read()
    
json_data = json.loads(json_string)

# Preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    #remove html tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    pattern_http = r'https://\S+'
    text = re.sub(pattern_http,'',text)
    pattern = r'\brt\b.*?:\s*' # this is to remove RT and and user name.
    text = re.sub(pattern,'',text)
    special_chars = re.escape(string.punctuation) # get all the possible list of special chars
    text = re.sub(r'\\u[\da-fA-F]{4}', '', text)
    text = re.sub(r'['+special_chars+']','',text) # replaing special chars
    text = re.sub(r'[^\w\s]','',text) # replace the punctuations


    #removal of numbers
    text = re.sub(r'[\d]','',text)

     #remove contractions
    text = contractions.fix(text)
    
    # Convert emojis to text
    text = emoji.demojize(text)

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



def process_chunk(chunk):
    return [preprocess_text(line) for line in chunk]



if __name__=='__main__':
    # Input and output file paths
    input_file_path = os.path.join(data_path,'data_text','output_twitter_data.txt')  # Update with your input file path
    output_file_path = os.path.join(data_path,'preprocessed_data','preprocessed_data_twitter.txt')  # Update with your output file path

    # Process each record in the file using multiprocessing and batch processing
    chunk_size = 1000

    # Process each record in the file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            lines = input_file.readlines()
            pool = Pool(processes=cpu_count())  # Use all available CPU cores
            chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
            preprocessed_chunks = pool.map(process_chunk, chunks)
            for chunk in preprocessed_chunks:
                output_file.writelines(chunk)
            
            # Handle any remaining lines
            remaining_lines = len(lines) % chunk_size
            if remaining_lines:
                remaining_chunk = lines[-remaining_lines:]
                remaining_preprocessed_chunk = process_chunk(remaining_chunk)
                output_file.writelines(remaining_preprocessed_chunk)

    pool.close()
    pool.join()

            
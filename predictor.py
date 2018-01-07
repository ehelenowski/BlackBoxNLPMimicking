import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import model_from_json
import html.parser
import csv
import string
import re
from sklearn.feature_extraction.text import CountVectorizer

seq_length = 15
MAX_SEQ_LENGTH = 15

cv = pickle.load(open("vocab.p", "rb"))
seq = np.zeros(seq_length)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

translator = str.maketrans('', '', string.punctuation)
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       "]+", flags=re.UNICODE)
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
html_parser = html.parser.HTMLParser()

def clean_string( string_to_clean ):
    #Remove links
    tweet = pattern.sub('', string_to_clean)
    #Remove Emojis
    tweet = emoji_pattern.sub(r'', tweet)
    #Escape HTML characters
    tweet = html_parser.unescape(tweet)
    #Remove @users and #s
    tweet = re.sub(r'\d*@\d*', '', tweet)
    tweet = re.sub(r'\d*#\d*', '', tweet)
    #Remove any numbers
    tweet = re.sub(r'\d+', '', tweet)
    #Remove punctuation
    tweet = tweet.translate(translator)
    #Remove any newlines or tabs
    tweet = tweet.lower().strip().replace('\n','').replace('\t', ' ').replace('-', '')
    tweet = tweet.replace("\"", '').replace("\”", '').replace('“','').replace('”', '')
    tweet = tweet.replace('–', '').replace("’",'').replace('   ', '').replace("—", ' ').replace(".", '')
    return tweet

def sequence_sentence( sentence ):
    word_array = sentence.split()
    seq = []
    count = 0
    for word in word_array:
        count += 1
        word_rep = cv.vocabulary_.get(word)
        if word_rep == None:
            word_rep = 0
        seq.append(word_rep)
        if count == MAX_SEQ_LENGTH:
            break
    if count < MAX_SEQ_LENGTH:
        for i in range(count+1,MAX_SEQ_LENGTH):
            seq.append(0)
    return seq


while True:
    sentence = input("Enter a sentence:\n")
    if sentence=="quit" or sentence=="q":
        break
    sentence = clean_string(sentence)
    print("The clean sentence is: ", sentence)
    seq = sequence_sentence(sentence)
    print("The resulting sequence is:\n",seq)
    list_o_list = []
    list_o_list.append(seq)
    seq = sequence.pad_sequences(list_o_list, maxlen=MAX_SEQ_LENGTH)
    prediction = model.predict(seq)[0]
    max_index = np.argmax(prediction)
    print("Predition is: ", prediction)
    if max_index == 0:
        print("That had a positive sentiment!\n")
    elif max_index == 1:
        print("That had a netural sentiment!\n")
    elif max_index == 2:
        print("That had a negative sentiment!\n")

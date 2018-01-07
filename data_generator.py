from sklearn.feature_extraction.text import CountVectorizer
import pickle
import html.parser
import csv
import string
import re

MAX_SEQ_LENGTH = 15

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

print("Gathering and cleaning data...")

true_lines = 0
start_at = 0
num_seen = 0

with open('trump_tweets_used', 'r') as fp:
    for line in fp:
        start_at = int(line)

print("Start at Trump tweet: ", start_at)

#Write the data into the appropriate files
with open('clean_data.txt', 'a') as cd:
    with open('raw_data.txt', 'a') as rd:
        with open('rawtwitterfeeds/DonaldTrumpTweets.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                true_lines += 1
                if true_lines >= start_at:
                    num_seen += 1
                    rd.write(row['text']+"\n")
                    tweet = clean_string(row['text'])
                    cd.write(tweet+"\n")
                if (num_seen == 1000):
                    break

next_start = start_at+num_seen

with open('trump_tweets_used', 'w+') as fp:
    fp.write(str(next_start))


print("Creating and saving the vocabulary...")

#Create our vocabulary
cv = CountVectorizer(max_features=20000,  stop_words='english')
X_train_counts = cv.fit_transform(tuple(open("clean_data.txt", 'r')))

#Save the vocabulary
pickle.dump(cv, open("vocab.p", "wb+"))

print("Generating sequence data...")

with open("sequence_data", "w") as sd:
    with open("clean_data.txt") as fp:
        for line in fp:
            word_array = line.split()
            count = 0
            for word in word_array:
                word_rep = cv.vocabulary_.get(word)
                count += 1
                if word_rep == None:
                    word_rep = 0
                if count < MAX_SEQ_LENGTH:
                    sd.write("{0} ".format(word_rep))
                elif count == MAX_SEQ_LENGTH:
                    sd.write("{0}\n".format(word_rep))
                    break
            if count < MAX_SEQ_LENGTH:
                for i in range(count,MAX_SEQ_LENGTH):
                    if i < MAX_SEQ_LENGTH-1:
                        sd.write("0 ")
                    else:
                        sd.write("0\n".format(word_rep))

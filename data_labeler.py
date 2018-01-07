from aylienapiclient import textapi
import time

app_id = ...
key = ...

client = textapi.Client(app_id, key)

true_counter = 0
num_seen = 0
start_at = 1

with open("raw_labels", 'r') as rl:
    for line in rl:
        start_at += 1

print("Labeling data points... (starting at {0})".format(start_at))

with open("raw_labels", 'a') as rl:
    with open("clean_data.txt", 'r') as rd:
        for line in rd:
            true_counter += 1
            if true_counter >= start_at:
                num_seen += 1
                print(true_counter)
                sentiment = client.Sentiment({'text':line})
                rl.write(sentiment['polarity']+"\n")
                time.sleep(1)
            if num_seen == 1000:
                break

print("Sequencing labels...")

true_counter = 1

with open("sequence_labels", "w") as sl:
    with open("raw_labels", "r") as rl:
        for line in rl:
            print(true_counter)
            true_counter += 1
            if line == 'positive\n':
                sl.write("1 0 0\n")
            elif line == 'neutral\n':
                sl.write("0 1 0\n")
            elif line == 'negative\n':
                sl.write("0 0 1\n")

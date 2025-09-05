import numpy as np
import nltk
import re
import string

from nltk.corpus import twitter_samples
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')
nltk.download('stopwords')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = positive_tweets[4000:]
train_pos = positive_tweets[:4000]
test_neg = negative_tweets[4000:]
train_neg = negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

print(f"Number of positive tweets: {len(positive_tweets)}")
print(f"Number of negative tweets: {len(negative_tweets)}")

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))


def process_tweet(tweet):
  stemmer = PorterStemmer()
  stopwords_english = stopwords.words('english')
  tweet = re.sub(r'\$\w*', '', tweet)
  tweet = re.sub(r'^RT[\s]+', '', tweet)
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
  tweet = re.sub(r'#', '', tweet)
  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
  tweet_tokens = tokenizer.tokenize(tweet)

  tweets_clean = []
  for word in tweet_tokens:
      if (word not in stopwords_english and
          word not in string.punctuation):
          stem_word = stemmer.stem(word)
          tweets_clean.append(stem_word)

  return tweets_clean


print("Before tweet processing: ", positive_tweets[0])
print("After tweet processing: ", process_tweet(positive_tweets[0]))

def build_freqs(tweets, ys):
  freq_dict = {}
  for tweet, y in zip(tweets, ys):
    tweet = process_tweet(tweet)
    for word in tweet:
      if (word, y) in freq_dict:
        freq_dict[(word, y)] += 1
      else:
        freq_dict[(word, y)] = 1
  return freq_dict

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

def train_naive_bayes(freq, train_x, train_y):
  vocab = set([pair[0] for pair in freq.keys()])
  V = len(vocab)
  loglikelihood = {}
  logprior = 0

  N_pos, N_neg = 0, 0
  V_pos, V_neg = 0, 0

  for pair in freq.keys():
    if pair[1] > 0.0:
      N_pos += freq[pair]
      V_pos += 1
    else:
      N_neg += freq[pair]
      V_pos += 1

  D = len(train_y)

  D_pos = len(list(filter(lambda x: x > 0, train_y)))
  D_neg = len(list(filter(lambda x: x <= 0, train_y)))

  logprior = np.log(D_pos) - np.log(D_neg)

  for word in vocab:
    freq_pos = freq.get((word, 1.0), 0)
    freq_neg = freq.get((word, 0.0), 0)

    temp_pos_prob = (freq_pos + 1) / (N_pos + V)
    temp_neg_prob = (freq_neg + 1) / (N_neg + V)

    loglikelihood[word] = np.log(temp_pos_prob / temp_neg_prob)

  return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)


def predict(tweet, logprior, loglikelihood):
  word_l = process_tweet(tweet)
  p = 0
  p += logprior
  for word in word_l:
    if word in loglikelihood:
      p += loglikelihood[word]
  return p

my_tweet = 'She smiled.'
p = predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

def evaluate(test_x, test_y, logprior, loglikelihood):
  accuracy = 0
  y_hats = []
  for tweet in test_x:
    y_hat = predict(tweet, logprior, loglikelihood)
    if y_hat > 0:
      y_hat_i = 1
    else:
      y_hat_i = 0
    y_hats.append(y_hat_i)
  accuracy = np.absolute(np.mean(np.equal(test_y, y_hats)))
  return accuracy

print("Naive Bayes accuracy = %0.4f" %
      (evaluate(test_x, test_y, logprior, loglikelihood)))

def predict_sentiment(tweet):
  p = predict(tweet, logprior, loglikelihood)

  if p > 1:
    return "Positive"
  elif p >= 0 and p <= 1:
    return "Neutral"
  else:
    return "Negative"
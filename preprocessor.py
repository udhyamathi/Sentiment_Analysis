import nltk
import re

nltk.download('stopwords')  # Stop words for english
nltk.download('wordnet')   # Vocabulary for lemmatization
nltk.download('vader_lexicon')   # lexicon for VADER model
nltk.download('punkt')   # Tokenizer rules
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def process_review(review):
    # Convert reviews into lowercase form
    review_lower=review.lower()

    # Remove punctuations, numbers
    review_puncs_nums_removed=re.sub(r"[^a-z\s]", "", review_lower)

    # Tokenize the review: convert sentence into words
    review_tokens=word_tokenize(review_puncs_nums_removed)

    # Remove stop words
    stop_words=stopwords.words('English')
    review_tokens_stopwords_removed=[token for token in review_tokens if token not in stop_words]

    # Lemmatization
    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens=[lemmatizer.lemmatize(token, pos='a') for token in review_tokens_stopwords_removed]

    # Put together all the list tokens as a single object
    return" ".join(lemmatized_tokens)

if __name__=="__main__":
    print(process_review("The product was really good. Timely delivered."))
    

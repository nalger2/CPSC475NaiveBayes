import nltk 
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

def read_file(fname):
    '''
    Reads a file as a list
    '''
    fin = open(fname, 'r')
    word_list = [line.strip() for line in fin.readlines()] #gets rid of /ns
    fin.close()
    return word_list


def makeBagOfWords(review):
    '''
    Extract words for a review
    Throw out those that are not alphabetic as well as short frequent English words
    Add the result to bag
    At the end, bag will be the list of words in a certain category of review.
    Use NLTK functions
    '''
                  
    bag = []
    words = movie_reviews.words(review)  #list of words in a review 
    words = [word for word in words if word.isalpha()] #remove items witn non-alpha chars
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words] #stop words removed
    bag = bag + words
    return bag
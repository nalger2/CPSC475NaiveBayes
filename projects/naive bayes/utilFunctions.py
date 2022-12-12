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

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for label in labels:
        row = [0 for label in labels] #[0, 0, 0]
        matrix.append(row) #[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(labels)): #check true
        for j in range(len(labels)): #check predicted
            for n in range(len(y_true)): #iterate through number of samples
                if y_true[n] == labels[i] and y_pred[n] == labels[j]:
                    matrix[i][j] += 1
    return matrix

def print_binary_confusion_matrix_labels(matrix):
    """Returns positional labels for each location in a binary matrix, assuming pos label
    is at positions 0 and 0

    Args:
        matrix(list list int): confusion matrix for a classifier

    Returns:
        tp (int): true positives
        fp (int): false positives
        tn (int): true negatives
        fn (int): false negatives
    """
    tp = matrix[0][0]
    print("true positives TP:", tp)
    fp = matrix[1][0]
    print("false positives FP:", fp)
    tn = matrix[1][1]
    print("true negatives TN:", tn)
    fn = matrix[0][1]
    print("false negatives:", fn)
    return tp, fp, tn, fn

def calc_classifier_stats(tp, fp, tn, fn, predicted):
    """Simple function to compute accuracy, precision, and recall of a classifier

    args:
        tp, fp, tn, fn: true pos, false pos, true neg, false neg
        predictions: list of predictions
    returns:
        accuracy, precision, recall
    
    """
    accuracy = tp / len(predicted)
    precision = tp / (tp + fp)
    recall = recall = tp / (tp + fn)

    return accuracy, precision, recall


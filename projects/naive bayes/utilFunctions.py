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

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = [y_true[i] for i in range(len(y_pred)) if y_true[i] == y_pred[i]]

    if correct and normalize:
            return float(len(correct) / len(y_pred))
    
    return len(correct)

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    
    if labels is None:
        labels = []
        temp_labels = set(y_true)
        for l in temp_labels:
            labels.append(temp_labels)

    if pos_label is None:
        pos_label = labels[0]

    tp_count = 0
    fp_count = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_pred[i] == pos_label:
            tp_count += 1
        elif y_pred[i] == pos_label:
            fp_count += 1

    try:
        precision = tp_count / (tp_count + fp_count)
    except:
        precision = 0
    
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = []
        temp_labels = set(y_true)
        for l in temp_labels:
            labels.append(temp_labels)

    if pos_label is None:
        pos_label = labels[0]

    tp_count = 0
    fn_count = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_pred[i] == pos_label:
            tp_count += 1
        elif y_pred[i] != pos_label and y_true[i] == pos_label:
            fn_count += 1

    try:
        recall = tp_count / (tp_count + fn_count)
    except:
        recall = 0

    return recall

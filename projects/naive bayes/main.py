import utilFunctions as utils
from NaiveBayesClassr import NaiveBayesClassifier


def main():
    #read train files
    pos_x_train = utils.read_file("pos_train_sample.txt") #TODO Replace these smaller file names with actual training data
    neg_x_train = utils.read_file("neg_train_sample.txt")
    #pos_x_train = utils.read_file("pos.txt") 
    #neg_x_train = utils.read_file("neg.txt")


    #create classifier
    nb_clf = NaiveBayesClassifier()
    
    #fit to train data
    priors, posteriors, vocabulary = nb_clf.fit(pos_x_train, neg_x_train)
    
    #read test files
    x_test = []
    #append first 100 positive test instances word bags
    pos_x_test_files = utils.read_file("posTst.txt")
    for filename in pos_x_test_files:
        x_test.append(utils.makeBagOfWords(filename))
    
    #append 2nd 100 negative test instances word bags
    neg_x_test_files = utils.read_file("negTst.txt")
    for filename in neg_x_test_files:
        x_test.append(utils.makeBagOfWords(filename))

    #test classifier
    predicted = nb_clf.predict(x_test, priors, posteriors, vocabulary)
    print("\n\nPREDICTIONS:\n", predicted)

    #write predictions to a file:
    with open("predictions.txt",'w') as fout:
        for item in predicted:
            fout.write(item)
            fout.write("\n")
    fout.close()

    #we know first 100 are positive, 2nd 100 are negative (gold standard)

    #TODO: make confusion matrix
    #TODO compute precision, recall, accuracy



if __name__ == "__main__":
    main()
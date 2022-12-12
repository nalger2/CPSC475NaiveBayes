import math

class NaiveBayesClassifier:
    
    def fit(self, x_train_pos, x_train_neg):
        print("fitting classifier to training data...")
        num_train_instances = len(x_train_pos) + len(x_train_neg)
        
        #compute priors
        priors = [math.log(len(x_train_pos)/num_train_instances), math.log(len(x_train_neg)/num_train_instances)]
                
        vocabulary = x_train_pos + x_train_neg #all vocab words in training set
        print("computing posteriors...")
        posteriors = {} #word : [pos likelihood, neg likelihood]
        numpos = len(x_train_pos)
        numneg = len(x_train_neg)
        i = 0 #just to print progress to output
        print(len(vocabulary))
        print(len(set(vocabulary)))
        
        #COMPUTE POSTERIORS:
        for word in set(vocabulary): #skips duplicates
            if i%1000 == 0:
                print(i, "/37k words trained")
            posteriors[word] = [0, 0]
            posteriors[word][0] = (math.log((x_train_pos.count(word) + 1) / numpos)) #pos likelihood (+1 to avoid 0 numerator)
            posteriors[word][1] = (math.log((x_train_neg.count(word) + 1) / numneg)) #neg likelhood (+1 to avoid 0 numerator)
            i = i+1
        return priors, posteriors, vocabulary
        
    def predict(self, x_test, priors, posteriors, vocabulary):
        print("testing classifier...")
        predicted = [] #list of classes predicted
        prob_positive = 0 #probability a word is classified as positive
        prob_negative = 0
        
        for test_instance in x_test:
            prob_positive = priors[0] #prior of class positive
            prob_negative = priors[1]
            
            for word in test_instance:
                if word in vocabulary: #skips words not in training vocabulary
                    prob_positive *= posteriors[word][0] #* posterior of word as class positive
                    prob_negative *= posteriors[word][1]
            
            #append corresponding classifier
            if prob_positive > prob_negative:
                predicted.append("pos")
            elif prob_negative > prob_positive:
                predicted.append("neg")

            #if probability is the same, predict the class with higher prior
            if prob_negative == prob_positive: 
                if priors[0] > priors[1]:
                    predicted.append("pos*")
                else:
                    predicted.append("neg")

        return predicted


        


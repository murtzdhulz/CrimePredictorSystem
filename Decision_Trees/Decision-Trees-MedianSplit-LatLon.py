class DecisionNode():

    def __init__(self, left, right, decision_function,class_label=None):
        self.left = left
        self.right = right
        self.decision_function = decision_make
        self.id = 0
        self.split_point = 0

        if not(isinstance(decision_function,list)):
            self.decision_function = decision_function
        elif len(decision_function) == 1:
            self.decision_function = decision_function[0]
        elif len(decision_function) == 2:
            self.decision_function = decision_function[0]
            self.id = decision_function[1]
        elif len(decision_function) == 3:
            self.decision_function = decision_function[0]
            self.id = decision_function[1]
            self.split_point = decision_function[2]

        if self.split_point is None:
            self.split_point = 0
        self.class_label = class_label

    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label
        
        return self.left.decide(feature) if self.decision_function(feature,self.id,self.split_point) else self.right.decide(feature)

# Constructing nodes one at a time,
# build a decision tree as specified above.
# There exists a correct tree with less than 6 nodes.

def decision_make(features,id,split_point):
    return features[id] <= split_point

#The decision_function parameter takes the following values: [decision_func,id,split_point]
class0 = DecisionNode(None,None,[None,None,None],0)
class1 = DecisionNode(None,None,[None,None,None],1)

def confusion_matrix(classifier_output, true_labels):
    #TODO output should be [[true_positive, false_negative], [false_positive, true_negative]]
    confusion_mx = [[0,0],[0,0]]
    for i in range(0,len(classifier_output)):
        if classifier_output[i] == true_labels[i]:
            if true_labels[i] == 1:
                confusion_mx[0][0] += 1
            else:
                confusion_mx[1][1] += 1
        else:
            if true_labels[i] == 1:
                confusion_mx[0][1] += 1
            else:
                confusion_mx[1][0] += 1
    return confusion_mx

def precision(classifier_output, true_labels):
    #TODO precision is measured as: true_positive/ (true_positive + false_positive)
    confusion_mx = confusion_matrix(classifier_output, true_labels)

    if confusion_mx[0][0]==0 and confusion_mx[1][0]==0:
        return 1.0

    precision = confusion_mx[0][0]/float(confusion_mx[0][0]+confusion_mx[1][0])
    return precision

def recall(classifier_output, true_labels):
    #TODO: recall is measured as: true_positive/ (true_positive + false_negative)
    confusion_mx = confusion_matrix(classifier_output, true_labels)

    if confusion_mx[0][0] == 0 and confusion_mx[0][1] == 0:
        return 1.0

    recall = confusion_mx[0][0]/float(confusion_mx[0][0]+confusion_mx[0][1])
    return recall

def accuracy(classifier_output, true_labels):
    #TODO accuracy is measured as:  correct_classifications / total_number_examples
    confusion_mx = confusion_matrix(classifier_output, true_labels)
    accuracy = (confusion_mx[0][0]+confusion_mx[1][1])/float(len(classifier_output))
    return accuracy

import numpy as np
from math import log

def entropy(class_vector):
    # TODO: Compute the Shannon entropy for a vector of classes
    # Note: Classes will be given as either a 0 or a 1.
    count_0 = class_vector.count(0)
    count_1 = class_vector.count(1)

    if count_0 == 0 or count_1 == 0:
        return 0

    entropy = -((count_0)/float(len(class_vector)))*log(((count_0)/float(len(class_vector))),2)-((count_1)/float(len(class_vector)))*log(((count_1)/float(len(class_vector))),2)
    return entropy

def information_gain(previous_classes, current_classes):
    # TODO: Implement information gain
    Hs = entropy(previous_classes)
    Hsv_0 = entropy(current_classes[0])
    Hsv_1 = entropy(current_classes[1])

    info_gain = Hs - len(current_classes[0])/float(len(previous_classes))*Hsv_0 - len(current_classes[1])/float(len(previous_classes))*Hsv_1

    return info_gain

#print information_gain([1,1,1,1,1,0,0,0],[[1,1,1,1,1],[0,0,0]])
#exit(0)

class DecisionTree():

    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        #TODO Implement the algorithm as specified above
        chosen_feature = self.build_sub_tree(features,classes,depth,[])
        return chosen_feature


    def build_sub_tree(self, features, classes, depth=0, traversed_features=[]):
        depth_cur = depth
        traversed_branch = list(traversed_features)
        count_0 = classes.count(0)
        count_1 = classes.count(1)

        if count_0 == 0:
            #print "Created leaf 1"
            return DecisionNode(None,None,[None,None,None],1)
        elif count_1 == 0:
            #print "Created leaf 0"
            return DecisionNode(None,None,[None,None,None],0)
        if depth>=self.depth_limit:
            #print 'Ceating a node coz length reached'
            if count_0 > count_1:
                return  DecisionNode(None,None,[None,None,None],0)
            return DecisionNode(None,None,[None,None,None],1)


        max_gain = float('-inf')
        best_feature = None
        best_left = []
        best_right = []
        best_features_left = []
        best_features_right = []
        best_split_point = 0
        medians = np.median(features,axis=0)

        for i in range(len(features[0])):
            if i in traversed_branch:
                #print 'SKIPPING',i
                continue
            features_left = []
            features_right = []
            list_left = []
            list_right = []

            cur_split_point = medians[i]

            for j in range(len(features)):
                if features[j][i] <= cur_split_point:
                    list_left.append(classes[j])
                    features_left.append(features[j])
                elif features[j][i] > cur_split_point:
                    list_right.append(classes[j])
                    features_right.append(features[j])

            info_gain = information_gain(classes,[list_left,list_right])
            #print 'Info gain for',i
            #print info_gain
            if info_gain > max_gain:
                #print 'Improved info gain is',info_gain
                max_gain = info_gain
                best_feature = i
                best_left = list_left
                best_right = list_right
                best_features_left = features_left
                best_features_right = features_right
                best_split_point = cur_split_point

        #print 'Best feature in build_tree=',best_feature
        traversed_branch.append(best_feature)
        depth_cur = depth_cur + 1
        #print 'Going for left'
        best_left_node = self.build_sub_tree(best_features_left,best_left,depth_cur,traversed_branch)
        #print 'Going for right'
        best_right_node = self.build_sub_tree(best_features_right,best_right,depth_cur,traversed_branch)
        #print 'Created new node for ',best_feature
        chosen_feature = DecisionNode(best_left_node,best_right_node,[decision_make,best_feature,best_split_point])
        return chosen_feature


    def classify(self, features):
        #TODO Use a fitted tree to classify a list of feature vectors
        # Your output should be a list of class labels (either 0 or 1)
        return [self.root.decide(feature) for feature in features]

def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    del rows[0]
    out = [  [float(i) for i in r.split(',')] for r in rows if r ]
    classes = []
    features = []
    for row in out:
      classes.append(int(row[class_index]))
      features.append(row[:class_index])
    return features, classes

import random
def generate_k_folds(dataset, k):
    #TODO this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    features = dataset[0]
    classes = dataset[1]

    rand_num = random.randint(1,100)

    random.seed(rand_num)
    random.shuffle(features)

    random.seed(rand_num)
    random.shuffle(classes)

    part_length = len(features)/k

    k_lists_features = []
    k_lists_classes = []
    list_fold = []

    for i in xrange(0,len(features),part_length):
        if (i+part_length) > len(features):
            k_lists_features.append(features[i:len(features)])
            k_lists_classes.append(classes[i:len(features)])
        else:
            k_lists_features.append(features[i:i+part_length])
            k_lists_classes.append(classes[i:i+part_length])

    for i in range(k):
        test_set = (k_lists_features[i],k_lists_classes[i])
        training_set_list_features = []
        training_set_list_classes = []
        for j in range(k):
            if j==i:
                continue
            training_set_list_features += k_lists_features[j]
            training_set_list_classes += k_lists_classes[j]
        training_set = (training_set_list_features,training_set_list_classes)
        list_fold.append((training_set,test_set))

    return list_fold

list_of_files = ['Murtaza2/ARSON.csv','Murtaza2/ASSAULT.csv','Murtaza2/BADCHECKS.csv','Murtaza2/BRIBERY.csv',
                 'Murtaza2/BURGLARY.csv','Murtaza2/DISORDERLYCONDUCT.csv','Murtaza2/DRIVINGUNDERTHEINFLUENCE.csv','Murtaza2/DRUGNARCOTIC.csv',
                 'Murtaza2/DRUNKENNESS.csv','Murtaza2/EMBEZZLEMENT.csv','Murtaza2/EXTORTION.csv','Murtaza2/FAMILYOFFENSES.csv',
                 'Murtaza2/FORGERYCOUNTERFEITING.csv','Murtaza2/FRAUD.csv','Murtaza2/GAMBLING.csv','Murtaza2/KIDNAPPING.csv',
                 'Murtaza2/LARCENYTHEFT.csv','Murtaza2/LIQUORLAWS.csv','Murtaza2/MISSINGPERSON.csv','Murtaza2/NONCRIMINAL.csv',
                 'Murtaza2/OTHEROFFENSES.csv','Murtaza2/PORNOGRAPHYOBSCENEMAT.csv','Murtaza2/PROSTITUTION.csv','Murtaza2/RECOVEREDVEHICLE.csv',
                 'Murtaza2/ROBBERY.csv','Murtaza2/RUNAWAY.csv','Murtaza2/SECONDARYCODES.csv','Murtaza2/SEXOFFENSESFORCIBLE.csv',
                 'Murtaza2/SEXOFFENSESNONFORCIBLE.csv','Murtaza2/STOLENPROPERTY.csv','Murtaza2/SUICIDE.csv','Murtaza2/SUSPICIOUSOCC.csv',
                 'Murtaza2/TREA.csv','Murtaza2/TRESPASS.csv','Murtaza2/VANDALISM.csv','Murtaza2/VEHICLETHEFT.csv',
                 'Murtaza2/WARRANTS.csv','Murtaza2/WEAPONLAWS.csv']

#on average your accuracy should be higher than 60%.
accuracies = []
precisions = []
recalls = []
confusion = []
sum_accuracy = 0

for oneFile in list_of_files:
    print "\n\nResult for file:",oneFile
    dataset = load_csv(oneFile)
    ten_folds = generate_k_folds(dataset, 10)
    iter_num = 0
    for fold in ten_folds:
        train, test = fold
        train_features, train_classes = train
        test_features, test_classes = test
        tree = DecisionTree(depth_limit=len(train_features[0]))
        tree.fit( train_features, train_classes)
        #print '########################Just before classifying:###################################',iter_num
        output = tree.classify(test_features)
        #print "###################################The accuracy at this stage is:#####################################",accuracy(output, test_classes),iter_num
        iter_num+=1
        accuracies.append( accuracy(output, test_classes))
        precisions.append( precision(output, test_classes))
        recalls.append( recall(output, test_classes))
        confusion.append( confusion_matrix(output, test_classes))

    sum_accuracy += np.mean(accuracies)
    print '\nID3: K-fold cross-validation results:'
    print 'Accuracies:',accuracies
    print 'Average accuacy for ID3 after k-fold validation is',np.mean(accuracies)
    print 'Precisions',precisions
    print 'Recalls',recalls
    print 'Confusion matrix',confusion

print 'Final big average accuracy is:',sum_accuracy/len(list_of_files)

exit(0)

#Random Forest Code
import random
class RandomForest():

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        # TODO implement the above algorithm to build a random forest of decision trees
        for i in range(0,self.num_trees):
            #Randomly select the data and the attributes. Randomly sampled.
            cur_features, cur_classes = self.get_features_classes(features,classes)
            cur_tree = DecisionTree(depth_limit=len(cur_features[0]))
            cur_tree.fit(cur_features,cur_classes)
            self.trees.append(cur_tree)

    #This helper function will return the reduced features and classes to be used for our random forest. Will be called in every iteration of tree generation.
    def get_features_classes(self,features,classes):
        no_of_attr = len(features[0]) * self.attr_subsample_rate
        no_of_attr = int(round(no_of_attr))
        no_of_examples = int(round(len(features) * self.example_subsample_rate))
        final_features = []
        final_classes = []

        attr_picked = random.sample(range(0,len(features[0])),no_of_attr)

        for i in range(0,no_of_examples):
            cur_index = random.randint(0,len(features)-1)
            final_classes.append(classes[cur_index])
            cur_feature = features[cur_index]
            new_feature = []
            for j in range(0,no_of_attr):
                new_feature.append(cur_feature[attr_picked[j]])
            final_features.append(new_feature)                      #This will be a list of lists

        return final_features,final_classes


    def classify(self, features):
        # TODO implement classification for a random forest.
        output_list = []
        for feature in features:
            results = []
            for i in range(0,self.num_trees):
                results.append(self.trees[i].root.decide(feature))
            if results.count(0) > results.count(1):
                output_list.append(0)
            else:
                output_list.append(1)
        return output_list

#TODO: As with the DecisionTree, evaluate the performance of your RandomForest on the dataset for part 2.
# on average your accuracy should be higher than 75%.

#  Optimize the parameters of your random forest for accuracy for a forest of 5 trees.
# (We'll verify these by training one of your RandomForest instances using these parameters
#  and checking the resulting accuracy)

#  Fill out the function below to reflect your answer:

def ideal_parameters():
    ideal_depth_limit = 3
    ideal_esr = 0.6
    ideal_asr = 0.75
    return ideal_depth_limit, ideal_esr, ideal_asr

accuracies_forest = []
precisions_forest = []
recalls_forest = []
confusion_forest = []

print '\nDoing the k-fold random forest:'
for fold in ten_folds:
    train, test = fold
    train_features, train_classes = train
    test_features, test_classes = test

    forest = RandomForest(5,3,0.6,0.75)
    forest.fit(train_features, train_classes)
    output = forest.classify(test_features)

    accuracies_forest.append( accuracy(output, test_classes))
    precisions_forest.append( precision(output, test_classes))
    recalls_forest.append( recall(output, test_classes))
    confusion_forest.append( confusion_matrix(output, test_classes))

print '\nRandom forest with k-fold cross validation results:'
print 'Accuracies',accuracies_forest
print 'Avg accuracy in Random Forest is',np.mean(accuracies_forest)
print 'Precisions',precisions_forest
print 'Recalls',recalls_forest
print 'Confusion matrix',confusion_forest
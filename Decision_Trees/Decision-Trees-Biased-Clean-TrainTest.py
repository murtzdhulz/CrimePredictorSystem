import csv

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
            #if i in traversed_branch:
                #print 'SKIPPING',i
                #continue
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

train_files = ['Murtaza-Train-Sets/ARSON.csv','Murtaza-Train-Sets/ASSAULT.csv','Murtaza-Train-Sets/BADCHECKS.csv','Murtaza-Train-Sets/BRIBERY.csv',
                'Murtaza-Train-Sets/BURGLARY.csv','Murtaza-Train-Sets/DISORDERLYCONDUCT.csv','Murtaza-Train-Sets/DRIVINGUNDERTHEINFLUENCE.csv','Murtaza-Train-Sets/DRUGNARCOTIC.csv',
                'Murtaza-Train-Sets/DRUNKENNESS.csv','Murtaza-Train-Sets/EMBEZZLEMENT.csv','Murtaza-Train-Sets/EXTORTION.csv','Murtaza-Train-Sets/FAMILYOFFENSES.csv',
                'Murtaza-Train-Sets/FORGERYCOUNTERFEITING.csv','Murtaza-Train-Sets/FRAUD.csv','Murtaza-Train-Sets/GAMBLING.csv','Murtaza-Train-Sets/KIDNAPPING.csv',
                'Murtaza-Train-Sets/LARCENYTHEFT.csv','Murtaza-Train-Sets/LIQUORLAWS.csv','Murtaza-Train-Sets/MISSINGPERSON.csv','Murtaza-Train-Sets/NONCRIMINAL.csv',
                'Murtaza-Train-Sets/OTHEROFFENSES.csv','Murtaza-Train-Sets/PORNOGRAPHYOBSCENEMAT.csv','Murtaza-Train-Sets/PROSTITUTION.csv','Murtaza-Train-Sets/RECOVEREDVEHICLE.csv',
                'Murtaza-Train-Sets/ROBBERY.csv','Murtaza-Train-Sets/RUNAWAY.csv','Murtaza-Train-Sets/SECONDARYCODES.csv','Murtaza-Train-Sets/SEXOFFENSESFORCIBLE.csv',
                'Murtaza-Train-Sets/SEXOFFENSESNONFORCIBLE.csv','Murtaza-Train-Sets/STOLENPROPERTY.csv','Murtaza-Train-Sets/SUICIDE.csv','Murtaza-Train-Sets/SUSPICIOUSOCC.csv',
                'Murtaza-Train-Sets/TREA.csv','Murtaza-Train-Sets/TRESPASS.csv','Murtaza-Train-Sets/VANDALISM.csv','Murtaza-Train-Sets/VEHICLETHEFT.csv',
                'Murtaza-Train-Sets/WARRANTS.csv','Murtaza-Train-Sets/WEAPONLAWS.csv']

list_of_files = ['Murtaza/ARSON.csv','Murtaza/ASSAULT.csv','Murtaza/BADCHECKS.csv','Murtaza/BRIBERY.csv',
                 'Murtaza/BURGLARY.csv','Murtaza/DISORDERLYCONDUCT.csv','Murtaza/DRIVINGUNDERTHEINFLUENCE.csv','Murtaza/DRUGNARCOTIC.csv',
                 'Murtaza/DRUNKENNESS.csv','Murtaza/EMBEZZLEMENT.csv','Murtaza/EXTORTION.csv','Murtaza/FAMILYOFFENSES.csv',
                 'Murtaza/FORGERYCOUNTERFEITING.csv','Murtaza/FRAUD.csv','Murtaza/GAMBLING.csv','Murtaza/KIDNAPPING.csv',
                 'Murtaza/LARCENYTHEFT.csv','Murtaza/LIQUORLAWS.csv','Murtaza/MISSINGPERSON.csv','Murtaza/NONCRIMINAL.csv',
                 'Murtaza/OTHEROFFENSES.csv','Murtaza/PORNOGRAPHYOBSCENEMAT.csv','Murtaza/PROSTITUTION.csv','Murtaza/RECOVEREDVEHICLE.csv',
                 'Murtaza/ROBBERY.csv','Murtaza/RUNAWAY.csv','Murtaza/SECONDARYCODES.csv','Murtaza/SEXOFFENSESFORCIBLE.csv',
                 'Murtaza/SEXOFFENSESNONFORCIBLE.csv','Murtaza/STOLENPROPERTY.csv','Murtaza/SUICIDE.csv','Murtaza/SUSPICIOUSOCC.csv',
                 'Murtaza/TREA.csv','Murtaza/TRESPASS.csv','Murtaza/VANDALISM.csv','Murtaza/VEHICLETHEFT.csv',
                 'Murtaza/WARRANTS.csv','Murtaza/WEAPONLAWS.csv']

output_list = []
for i in range(0,len(list_of_files)):
    cur_file = train_files[i]
    full_file = list_of_files[i]

    print "\n\nProcessing for the file:",full_file

    dataset = load_csv(cur_file)
    features = dataset[0]
    classes = dataset[1]
    tree = DecisionTree(depth_limit=20)
    tree.fit(features, classes)

    dataset2 = load_csv(full_file)
    features2 = dataset2[0]
    classes2 = dataset2[1]
    output = tree.classify(features2)

    output_list.append(output)

    print 'Accuracy is',accuracy(output, classes2)
    print 'Confusion matrix is:',confusion_matrix(output, classes2)

dataset2 = load_csv('Murtaza/ARSON.csv')
features2 = dataset2[0]
classes2 = dataset2[1]
#print 'Prediction for first ten features'
with open('Decision-Trees-Results-25thNovember-2015-2.csv','wb') as my_file:
    csv_wrie = csv.writer(my_file)
    csv_wrie.writerow(['ARSON','ASSAULT','BADCHECKS','BRIBERY',
                       'BURGLARY','DISORDERLYCONDUCT','DRIVINGUNDERTHEINFLUENCE','DRUGNARCOTIC',
                       'DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILYOFFENSES',
                       'FORGERYCOUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING',
                       'LARCENYTHEFT','LIQUORLAWS','MISSINGPERSON','NONCRIMINAL',
                       'OTHEROFFENSES','PORNOGRAPHYOBSCENEMAT','PROSTITUTION','RECOVEREDVEHICLE',
                       'ROBBERY','RUNAWAY','SECONDARYCODES','SEXOFFENSESFORCIBLE',
                       'SEXOFFENSESNONFORCIBLE','STOLENPROPERTY','SUICIDE','SUSPICIOUSOCC',
                       'TREA','TRESPASS','VANDALISM','VEHICLETHEFT',
                       'WARRANTS','WEAPONLAWS'
                       ])
    for iter in range(len(features2)):
        predict = []
        #print 'Checking for DataLine',iter
        for i in range(len(list_of_files)):
            predict.append(output_list[i][iter])
            #if output_list[i][iter] == 1:
                #print 'The crime predicted is:',list_of_files[i]
        #print 'Prediction Array is:'
        #print predict
        csv_wrie.writerow(predict)

#Here I will try to do for biased tree
print 'Just before exit'
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:02:53 2020

References
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
GitHub/Friends-Recommender-In-Social-Network-master/LinkPredictionInSocialNetwork.ipynb
"""

import networkx as nx
import random
import csv
import datetime
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import log_loss 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#%%

# Function to count common neighbours 

def common_neighbors(g, edges):
    result = []
    for edge in edges:
        node_one, node_two = edge[0], edge[1]
        num_common_neighbors = 0
        try:
            neighbors_one, neighbors_two = g.neighbors(node_one), g.neighbors(node_two)
            for neighbor in neighbors_one:
                if neighbor in neighbors_two:
                    num_common_neighbors += 1
            result.append((node_one, node_two, num_common_neighbors))
        except:
            pass
    return result

#%%
    
# Set features to run
feature_set = [common_neighbors,
                   nx.resource_allocation_index,
                   nx.jaccard_coefficient,
                   nx.adamic_adar_index,
                   nx.preferential_attachment
                   ]

#%%

# Produce fake edges to provide negative samples

def produce_fake_edge(g, neg_g,num_test_edges):
    i = 0
    while i < num_test_edges:
        edge = random.sample(g.nodes(), 2)
        try:
            shortest_path = nx.shortest_path_length(g,source=edge[0],target=edge[1])
            if shortest_path >= 2:
                neg_g.add_edge(edge[0],edge[1], positive="False")
                i += 1
        except:
            pass
#%%
# Create graph based on input file
            
def create_graph_from_file(filename):
    print("----------------build graph--------------------")
    f = open(filename, "rb")
    g = nx.read_edgelist(f)
    return g

#%%

# Extract positive and negative samples for training
    
def sample_extraction(g, pos_num, neg_num, neg_mode, neg_distance=2, delete=1):
    """

    :param g:  the graph
    :param pos_num: the number of positive samples
    :param neg_num: the number of negative samples
    :param neg_distance: the distance between two nodes in negative samples
    :param delete: if delete ==0, don't delete positive edges from graph
    :return: pos_sample is a list of positive edges, neg_sample is a list of negative edges
    """

    print("----------------extract positive samples--------------------")
    # randomly select pos_num as test edges
    pos_sample = random.sample(g.edges(), pos_num)
    sample_g = nx.Graph()
    sample_g.add_edges_from(pos_sample, positive="True")
    nx.write_edgelist(sample_g, "sample_positive_" +str(pos_num)+ ".txt", data=['positive'])

    # adding non-existing 
    print("----------------extract negative samples--------------------")
    #i = 0
    neg_g = nx.Graph()
    produce_fake_edge(g,neg_g,neg_num)
    nx.write_edgelist(neg_g, "sample_negative_" +str(neg_num)+ ".txt", data=["positive"])
    neg_sample = neg_g.edges()
    neg_g.add_edges_from(pos_sample,positive="True")
    nx.write_edgelist(neg_g, "sample_combine_" +str(pos_num + neg_num)+ ".txt", data=["positive"])

    # remove the positive sample edges
    if delete == 0:
        return pos_sample, neg_sample
    else:
        g.remove_edges_from(pos_sample)
        nx.write_edgelist(g, "training.txt", data=False)

    return pos_sample, neg_sample
     

#%%
# Produce topologica attributes from link prediction algorithms for training supervised ML classifiers
    
def feature_extraction(g, pos_sample, neg_sample, feature_name, model, combine_num):

    data = []
    if model == "single":
        print ("-----extract feature:", feature_name.__name__, "----------")
        preds = feature_name(g, pos_sample)
        feature = [feature_name.__name__] + [i[2] for i in preds]
        label = ["label"] + ["Pos" for i in range(len(feature))]
        preds = feature_name(g, neg_sample)
        feature1 = [i[2] for i in preds]
        feature = feature + feature1
        label = label + ["Neg" for i in range(len(feature1))]
        data = [feature, label]
        data = transpose(data)
        print("----------write the feature to file---------------")
        write_data_to_file(data, "features_" + model + "_" + feature_name.__name__ + ".csv")
    else:
        label = ["label"] + ["1" for i in range(len(pos_sample))] + ["0" for i in range(len(neg_sample))]
        for j in feature_name:
            print ("-----extract feature:", j.__name__, "----------")
            preds = j(g, pos_sample)

            feature = [j.__name__] + [i[2] for i in preds]
            preds = j(g, neg_sample)
            feature = feature + [i[2] for i in preds]
            data.append(feature)

        data.append(label)
        data = transpose(data)
        print("----------write the features to file---------------")
        write_data_to_file(data, "features_" + model + "_" + str(combine_num) + ".csv")
    return data


def write_data_to_file(data, filename):
    csvfile = open(filename, "w")
    writer = csv.writer(csvfile)
    for i in data:
        writer.writerow(i)
    csvfile.close()


def transpose(data):
    return [list(i) for i in zip(*data)]


#%%
# Main function to kick off the process
        
def main(filename, pos_num, neg_num, model, combine_num, feature_name, neg_mode):
   
    g = create_graph_from_file(filename)
    num_edges = g.number_of_edges()
    pos_num = int(num_edges * pos_num)
    neg_num = int(num_edges * neg_num)
    pos_sample, neg_sample = sample_extraction(g, pos_num, neg_num,neg_mode)
    train_data = feature_extraction(g, pos_sample, neg_sample, feature_name, model, combine_num)    
    
#%%

# Define variables for calling the main function
    
fn="Amazon0302.txt";
cn=9;
pos =0.05;
neg =0.05;
mode="easy";
f_model = "combined"; 
      

#%%
# Call the main function

main(filename=fn,model=f_model,pos_num=pos, neg_num=neg, combine_num=cn, feature_name=feature_set, neg_mode=mode)

#%%
# Perform train-test split on sampled graph after featurization 

r=np.loadtxt(open("features_combined_"+str(cn)+".csv", "rb"), delimiter=",", skiprows=1);

l,b=r.shape;

np.random.shuffle(r);

pct = 0.70

train_l=int(pct*l)
X_train=r[0:train_l,0:b-1]
Y_train=r[0:train_l,b-1]
X_test=r[train_l:l,0:b-1]
Y_test=r[train_l:l,b-1]

#Feature Scaling
# standardize the data prior to fitting the model
# so that the features will have the properties of a standard normal distribution 

X_train = normalize(X_train, axis=0, norm='max')
X_test = normalize(X_test, axis=0, norm='max')
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


#%%

# Function to run Logistic Classification 

def logistic(training, training_labels, testing, testing_labels):
    clf = LogisticRegression()
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    result=clf.predict(testing)
    np.set_printoptions(precision=2)
    print ("+++++++++ Finishing training the Logistic classifier ++++++++++++")
    print ("Logistic Classfication accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ('Process time:', (finish-start).seconds)
    


#%%

# Function to run Decision Tree Classification
def decisiontree(training, training_labels, testing, testing_labels):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    result=clf.predict(testing)
    
    print ("+++++++++ Finishing training the Decision Tree classifier ++++++++++++")
    print ("Decision Tree Classfication accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ('Process time:', (finish-start).seconds)
    
#%%
#Function to run KNeighborsClassifier Classification

def kNN(training, training_labels, testing, testing_labels):
    clf = KNeighborsClassifier()
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the KNeighbors Classifier ++++++++++++")
    result = clf.predict(testing)
    print ("KNeighbors Classifier accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ((finish-start).seconds)
    

#%%
#Function to run RandomForestClassifier Classification

def RFC(training, training_labels, testing, testing_labels):
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the RandomForest Classifier ++++++++++++")
    result = clf.predict(testing)
    print ("RandomForest Classifier accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ((finish-start).seconds)
    

#%%
#Function to run AdaBoostClassifier Classification

def AdB(training, training_labels, testing, testing_labels):
    clf = AdaBoostClassifier()
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the AdaBoost Classifier ++++++++++++")
    result = clf.predict(testing)
    print ("AdaBoost Classifier accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ((finish-start).seconds)
    
    

#%%
#Function to run GaussianNB Classification

def GNB(training, training_labels, testing, testing_labels):
    clf = GaussianNB()
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the GaussianNB Classifier ++++++++++++")
    result = clf.predict(testing)

    print ("GaussianNB Classifier accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ((finish-start).seconds)


#%%
# Function to run SVM

def Svm(training, training_labels, testing, testing_labels):
    #Support Vector Machine
    start = datetime.datetime.now()
    clf = svm.SVC()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the SVM classifier ++++++++++++")
    result = clf.predict(testing)
    print ("SVM accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score = 2 * (precision * recall) / (precision + recall):", f1_score(testing_labels, result))
    print ("Precision Score = tp / (tp + fp):", precision_score(testing_labels, result))
    print ("Recall Score = tp / (tp + fn):", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score:'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ('Process time:', (finish-start).seconds)


#%%
#Function to run Neural Net Classification

def ANN(training, training_labels, testing, testing_labels):
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15,9), random_state=1)
    #clf = MLPClassifier(alpha=1, max_iter=1000)
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the ANN classifier ++++++++++++")
    result = clf.predict(testing)
    print ("ANN accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ((finish-start).seconds)
    

#%%

#Function to run Quadratic Discriminant Analysis  

def QDA(training, training_labels, testing, testing_labels):
    clf = QuadraticDiscriminantAnalysis()
    start = datetime.datetime.now()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the Quadratic Discriminant Analysis ++++++++++++")
    result = clf.predict(testing)
    print ("QDA accuracy:", accuracy_score(testing_labels, result))
    print ("Confusion Matrix:", confusion_matrix(testing_labels, result))
    # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #              ("Normalized confusion matrix", 'true')]
    # for title, normalized in titles_options:
    #     disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize=normalized)
    #     disp1.ax_.set_title(title)
    disp1 = plot_confusion_matrix(clf, X_test, Y_test, cmap= plt.cm.Blues, normalize='true')
    disp1.ax_.set_title("Normalized confusion matrix")
    print ("F1 Score:", f1_score(testing_labels, result))
    print ("Precision Score:", precision_score(testing_labels, result))
    print ("Recall Score:", recall_score(testing_labels, result))
    print ("Classification report:")
    print (classification_report(testing_labels, result))
    print ("ROC Score:", roc_auc_score(testing_labels, result))
    print ("Log Loss Score:", log_loss(testing_labels, result, eps=1e-15))
    average_precision = average_precision_score(testing_labels, result)
    print('Average precision-recall score'.format(average_precision))
    disp = plot_precision_recall_curve(clf, X_test, Y_test)
    disp.ax_.set_title('2-class Precision-Recall curve')
    #keep the time
    finish = datetime.datetime.now()
    print ((finish-start).seconds)
    
    
#%%

#Run this for Logistic Regression
logistic(X_train,Y_train,X_test,Y_test)
#Run this for Decision Tree classification
decisiontree(X_train,Y_train,X_test,Y_test)
# Run this for kNN classification
GNB(X_train,Y_train,X_test,Y_test)
# Run this for kNN classification
AdB(X_train,Y_train,X_test,Y_test)
# Run this for kNN classification
RFC(X_train,Y_train,X_test,Y_test)
# Run this for kNN classification
kNN(X_train,Y_train,X_test,Y_test)
# Run this for ANN classification
ANN(X_train,Y_train,X_test,Y_test)
# Run this for QDA classification
QDA(X_train,Y_train,X_test,Y_test)
#Run this to for SVM classification
Svm(X_train,Y_train,X_test,Y_test)





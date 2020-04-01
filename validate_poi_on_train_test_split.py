
# coding: utf-8

# In[1]:

import pickle, sys, os


# In[2]:

sys.path.append("C:\\Users\\THB4UT\\Downloads\\ud120-projects-master\\tools"); sys.path


# In[3]:

from feature_format import featureFormat, targetFeatureSplit


# In[4]:

data_dict = pickle.load(open("..\\final_project\\final_project_dataset.pkl", "rb"))


# In[5]:

features_list = ["poi", "salary"]


# In[6]:

data = featureFormat(data_dict, features_list)


# In[7]:

labels, features = targetFeatureSplit(data)


# In[8]:

# from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)


# In[9]:

from sklearn.model_selection import KFold


# In[10]:

# Kfold
kf = KFold(10, shuffle = True)
for train_indices, test_indices in kf.split(features):
    #make trainitn and testing datasets
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]


# In[11]:

# Unit 17: Create a Decision Tree Classifier with Default parameters
from sklearn import tree
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[12]:

# Create the Classifier
parameters = {'criterion' : ('gini', 'entropy'), 
              'splitter' : ('best','random'),
             'min_samples_split' : [2, 5, 10, 15, 20, 30, 40, 50],
             'max_depth' : [1, 2, 3, 4, 5, 6, 7]}
dt = tree.DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)


# In[13]:

# start the timer and train the classifier on all data points
t0 = time()
#clf = clf.fit(features, labels)               # Unit 17
clf = clf.fit(features_train, labels_train)   # Unit 18
print("Training: ", round(time()-t0, 3), "s")


# In[14]:

# make predictions
t0 = time()
#pred = clf.predict(features)              # Unit 17
pred = clf.predict(features_test)         # Unit 18
print("Predicting: ", round(time()-t0, 3), "s")


# In[15]:

# Print the accuracy
#acc = accuracy_score(pred, labels)             # Unit 17
acc = accuracy_score(pred, labels_test)        # Unit 18
print("accuracy: ", round(acc, 10))


# In[16]:

print("Best parameters: ", clf.best_params_)


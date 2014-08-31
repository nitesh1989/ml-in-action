from numpy import *
import operator


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,1.1]])
    labels = ['A','A','B','B']
    return group, labels

# Import the kNN library
import kNN

# Create groups and labels
groups,labels = kNN.createDataSet()

# Check groups and labels
print "groups: \n", groups
print "labels: \n", labels

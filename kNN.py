from sklearn import *
from numpy import *
import operator


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,1.1]])
    labels = ['A','A','B','B']
    return group, labels

# Import the kNN library
import kNN

# Create groups and labels
group,labels = kNN.createDataSet()

# Check groups and labels
print "group: \n", group
print "labels: \n", labels


# Algorithm Pseudo code
""" 
Data Set name: inX

For every point in our dataset:
    Calculate the distance between inX and the current point
    Sort the distances in increasing order
    Take K items with the lowest distance to inX
    Find the majority class amoung these k items
    return the majority class as the prediction for class of inX
"""


def classify0(inX, dataSet, labels, k):
    # Get number of rows in data set
    dataSetSize = dataSet.shape[0]
#    print "dataSet.shape: ", dataSet.shape
#    print "dataSet.__class__ :", dataSet.__class__
    # Get the difference matrix between inX and your dataSet
    # notice the use of numpy.tile
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
#    print "tile: ", tile(inX, (dataSetSize,1))
#    print "diffMat: ", diffMat
#    print "diffMar.__class__: ", diffMat.__class__
    sqDiffMat = diffMat**2
#    print "sqDiffMat: ", sqDiffMat
    sqDistances = sqDiffMat.sum(axis=1)
#    print "sqDistances: ", sqDistances
    distances = sqDistances**0.5
#    print "distances:  ",distances

    # Sort distances from smallest to largest
    sortedDistIndicies = distances.argsort()
#    print sortedDistIndicies
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
#        print "voteILabel: ", voteIlabel
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
#        print "classCount.get(voteIlabel,0) + 1: ", classCount.get(voteIlabel,0) + 1
#        print classCount[voteIlablel]
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    print sortedClassCount
    return sortedClassCount[0][0]

if __name__ == "__main__":
    kNN.classify0([0,0], group, labels, 3)
    kNN.classify0([0,1], group, labels, 3)

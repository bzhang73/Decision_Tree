#
#  Created by BO ZHANG on 9/13/19.
#  Copyright © 2019 BO ZHANG. All rights reserved.
#
import pandas as pd
import numpy as np
import math
import operator
import csv
import pickle
import sys

def createDataSet(dataname):
    dataSet = np.array(dataname)
    labels = list(dataname.columns.values)
               #change to discrete values
    return dataSet, labels


#calculate entropy
def calcEntropy(dataSet):
    numEntries = len(dataSet); #number of data
    labelCounts = {}  #key is the last row element value, value is the number of the data choosed
    
    for featVec in dataSet: #find all the dataSet
        currentLabel = featVec[-1] #the value of the last element
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0
        
        labelCounts[currentLabel]+=1
    
    shannonEntropy = 0.0  #intial the entropy
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEntropy -= prob * math.log(prob,2) #log base 2 calculate the entropy
    return shannonEntropy


def uniquecounts(rows):
    results = {}
    for row in rows:
        #The result is the last column
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
        return results

def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        # imp+=p1*p1
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp  # 1-imp

#split the data
#axis is the set in dataSet，value is the specific difference value of the set
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet: #find all dataSet
        if featVec[axis] == value: #
            reducedFeatVec = featVec[:axis]  #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    #print axis, value, reduced FeatVec
#    print(retDataSet)
    return retDataSet

#choose the dataSet and calculate the most value feature
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1 #find the number of the feature, the last featur is the class
    baseEntropy = calcEntropy(dataSet) # calculate the dataSet entropy
    bestInfoGain = 0.0;
    bestFeature =-1; #iterate over all the features
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #get all the feature in the dataSet
        uniqueVals = set(featList) #get current feature
        newEntrop = 0.0
        #calculate every featur information
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntrop += prob * calcEntropy(subDataSet)
        infoGain = baseEntropy - newEntrop #calculate the information
        if (infoGain>bestInfoGain):
            bestInfoGain = infoGain     #if better than current best, set to best
            bestFeature = i
    return bestFeature      #returns an integer


#choose the dataSet and calculate the most value feature using gini impurity
def chooseBestFeatureToSplit2(dataSet):
    numFeatures = len(dataSet[0])-1 #find the number of the feature, the last featur is the class
    baseGiniPurity = giniimpurity(dataSet) # calculate the dataSet entropy
    bestInfoGain = 1.0;
    bestFeature =-1; #iterate over all the features
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #get all the feature in the dataSet
        uniqueVals = set(featList) #get current feature
        newGiniPurity = 0.0
        #calculate every featur information
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newGiniPurity += prob * giniimpurity(subDataSet)
        infoGain = baseGiniPurity - newGiniPurity #calculate the information
        if (infoGain<bestInfoGain):
            bestInfoGain = infoGain     #if better than current best, set to best
            bestFeature = i
    return bestFeature      #returns an integer


#this function classification the name of the list, then cteate a classList diction
#classList diction store all the lable frequency
#finally using operator to sort the diction
#return the max frequency label
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)

##create a decision treee
def createTree(dataSet,labels,solution):
    classList = [example[-1] for example in dataSet] #return current label all value
    if classList.count(classList[0]) == len(classList):
        return classList[0]  #if the classList is the same, return this label
    if len(dataSet[0])==1:  #lierate over all the feature and cannot split the dataSet
        return majorityCnt #return the most frequency label
    if solution == 0:
        bestFeature = chooseBestFeatureToSplit(dataSet) #get the bestFeature
    else:
        bestFeature = chooseBestFeatureToSplit2(dataSet) #get the bestFeature
    bestFeatLabel = labels[bestFeature] #get the feature label
#return bestFeatLabel
#build the tree
    #store the inforemation of the tree
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeature]) #delete the current feature
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels,solution)  #splitDataSet
    return myTree


#classify the tree
#def classfy(inputTree,featLabels, testVec):
#    firstStr = inputTree.keys()[0]
#    secondDic = inputTree[firstStr]
#    featIndex = featLabels.index(firstStr)
#    key = testVec[featIndex]
#    valueOfFeature = secondDic[key]
#    if isinstance(valueOfFeature,dict):
#        classLabel = classify(valueOfFeature,featLabels,testVec)
#    else:
#        classLabel = valueOfFeature
#    return classLabel

def classify(inputTree,featLabels,testVector):
     root = list(inputTree.keys())[0] #the first label is the root
     dictionary = inputTree[root] #find the dictionary-
     featIndex = featLabels.index(root)
     classLabel =""
     for key in dictionary.keys():#iterator dictionary
         if testVector[featIndex] == key:
             if type(dictionary[key]).__name__ == 'dict': #if there another dictionary
                 classLabel = classify(dictionary[key],featLabels,testVector)
             #Top down recursion find non dictionary, then save the label
             else:
                 classLabel=dictionary[key]#the leafe return the label
     return classLabel

def test(myTree,labels,dataSet,sum,correct,error):
     for line in dataSet:
         result=classify(myTree,labels,line)
#         print(result)
#         print(line[-1])
#         print(type(result))
#         print(type(line[-1]))
#         exit(0)
         if result==line[-1]: #
             correct = correct + 1
         else :
             error = error + 1
#         print(correct)
#         print(error)
#         exit(0)
#     print(correct)
#     print(sum)
#     print(correct / sum)
#     print("accurace：%", (correct / sum)*100 )
     return sum,correct,error


def storetree(inputree,filename):
    fw = open(filename, 'wb')
    pickle.dump(inputree, fw)
    fw.close()



def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


#find the subdataset
# dataSet
# axis --col
# value --label
def splitSubDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            retDataSet.append([featVec[axis],featVec[-1]])
    return retDataSet


def postPruning(inputTree,dataSet,validateData,label):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    classList = [example[-1] for example in dataSet]
    featkey = firstStr
    labelIndex = label.index(featkey)
    temp_labels = label.copy()
    del (label[labelIndex])
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            if type(dataSet[0][labelIndex]).__name__ == 'str':
                inputTree[firstStr][key] = postPruning(secondDict[key], splitDataSet(dataSet, labelIndex, key),splitDataSet(validateData, labelIndex, key),label.copy())
            else:
                inputTree[firstStr][key] = postPruning(secondDict[key], splitDataSet(dataSet, labelIndex, key),splitDataSet(validateData, labelIndex,key), label.copy())
    beforeCorrect = undivideCorrect(classList, validateData)
    afterCorrect = divideCorrect(dataSet, secondDict.keys(), labelIndex, validateData)
    if (beforeCorrect > afterCorrect):
        return majorityCnt(classList)
    return inputTree

# uncorrect pruning
def undivideCorrect(classList,validateData):
#    print(len(validateData))
#    print(max(classList))
#    print("\n")
    if len(validateData):
        good = splitSubDataSet(validateData, len(validateData[0]) - 1, max(classList))  # the number of the correct
        beforeCorrect = len(good) / len(validateData)  # the accurace
        return beforeCorrect
    else:
        var=0
        return var

# correct pruning
def divideCorrect(dataSet,uniqueVals,bestFeat,validateData):
    good = 0
    for value in uniqueVals:  # iteritor all
        featList = [feat[-1] for feat in splitDataSet(dataSet, bestFeat, value)] # find result
        templList = splitSubDataSet(validateData, bestFeat, value)  # find the number
        goodList = []
        if(len(templList)>0):
            goodList = splitSubDataSet(templList, len(templList[0]) - 1, max(featList))  # the number of the correct
        good +=len(goodList)

        if len(validateData)==0:
            var=0;
            return var;
        else:
            return good / len(validateData)  # accuracy

#get the length of the tree
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]#'dict_keys' object does not support indexing
    print(firstStr)
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:numLeafs+=1
    return numLeafs


def testprint(a,cnt):
    for each_key in a:
        print('|' * cnt,end=' ')
        if type(a[each_key]) is dict:
            print(each_key)
            testprint(a[each_key],cnt+1)
        else:
            print(each_key,' -> ', a[each_key])

#myDat,labels= createDataSet()
#mytrees=createTree(myDat,labels)
#print(mytrees)

#using pd.read_csv  read the file, then using numpy to translate to matrix
#testdata = pd.read_csv("test_set.csv");
#mylabel =list(testdata.columns.values)
#print(mylabel)
#myDat,labels= createDataSet(testdata)

def main(argv):
    if len(sys.argv) !=6:
        print('Usage: python3 dectree.py <training-set> <validation-set> <test-set> <to-print> to_print:{yes,no} <prune> prune:{yes, no}')
        exit(0)
    print('the number of arguement: ', len(sys.argv))
    print('the list of arguement: ', str(sys.argv))
    print('my code name: ', sys.argv[0])
    for i in range(1,len(sys.argv)):
        print('arguement %s is: %s' %(i,sys.argv[i]))

    trainingFileName = sys.argv[1]
    validationFileName = sys.argv[2]
    testFileName = sys.argv[3]

    printTree = sys.argv[4]
    printPruneTree = sys.argv[5]
    if printTree != 'yes' and printTree!='no':
        print('Usage: python3 dectree.py <training-set> <validation-set> <test-set> <to-print> to_print:{yes,no} <prune> prune:{yes, no}')
        exit(0)

    if printPruneTree != 'yes' and printPruneTree!='no':
        print('Usage: python3 dectree.py <training-set> <validation-set> <test-set> <to-print> to_print:{yes,no} <prune> prune:{yes, no}')
        exit(0)


    with open(trainingFileName, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    myDat=your_list[1:]
    labels =  your_list[0][:]
    label1=your_list[0][:]
    label2=your_list[0][:]
    label3=your_list[0][:]
    label4=your_list[0][:]
    label5=your_list[0][:]
    label6=your_list[0][:]
#print(myDat)
#print(labels)
    mytrees=createTree(myDat,labels,0)

#print("print length of the tree\n")
#    length = getNumLeafs(mytrees)
#    print(length)
    if printTree == 'yes' :
        print()
        testprint(mytrees,1)
        print()
#print(mytrees)
#storetree(mytrees,'mytree.txt')
#print("\n");
#
#for key in mytrees.keys():
#    print(key, end="\n|")
    sum = len(myDat)
    correct=0
    error=0
    fw = open("results.txt", 'a', encoding='utf8')


#print(label1)
    print("\nThe First decision tree:")
    fw.write('The First decision tree:\n')
#fw.write('Hello, world!')
    sum,correct,error=test(mytrees,label1,myDat,sum,correct,error)
#print("\n")
    print("Training：", (correct / sum)*100, "%" )
    fw.write('Training：%.2f' % (correct / sum))
    fw.write("\n")

    with open(validationFileName, 'r') as f2:
        reader2 = csv.reader(f2)
        your_list2 = list(reader2)

    myValidatDat=your_list2[1:]
#print("\n")
    correct=0
    error=0
    sum,correct,error=test(mytrees,label1,myValidatDat,sum,correct,error)
#print("\n")
    print("Validation：", (correct / sum)*100, "%" )
    fw.write('Validation：%.2f' % (correct / sum))
    fw.write("\n")


    with open(testFileName, 'r') as f1:
        reader1 = csv.reader(f1)
        your_list1 = list(reader1)

    myTestDat=your_list1[1:]
#print("\n")
    correct=0
    error=0
    sum,correct,error=test(mytrees,label1,myTestDat,sum,correct,error)
#print("\n")
    print("Test：", (correct / sum)*100, "%" )
    fw.write('Test：%.2f' % (correct / sum))
    fw.write("\n")



    print("\nThe First decision tree with pruning:")
    fw.write('\nThe First decision tree with pruning:\n')
    postPruning(mytrees,myDat,myValidatDat,label1)

    if printPruneTree == 'yes' :
        print()
        testprint(mytrees,1)
        print()

#print(mytrees)

    correct=0
    error=0
    sum,correct,error=test(mytrees,label3,myDat,sum,correct,error)
    print("Training：", (correct / sum)*100, "%" )
    fw.write('Training：%.2f' % (correct / sum))
    fw.write("\n")

    correct=0
    error=0
    sum,correct,error=test(mytrees,label3,myValidatDat,sum,correct,error)
#print("\n")
    print("Validation：", (correct / sum)*100, "%" )
    fw.write('Validation：%.2f' % (correct / sum))
    fw.write("\n")

    correct=0
    error=0
    sum,correct,error=test(mytrees,label3,myTestDat,sum,correct,error)
#print("\n")
    print("Test：", (correct / sum)*100, "%" )
    fw.write('Test：%.2f' % (correct / sum))
    fw.write("\n")

#print("\n")
#print(myDat)
#print(labels)
#print(testdata.columns)
#print(testdata.loc[0])
#arr1 = np.array(testdata);
#print("\n\n array\n\n")
#print(arr1)
#print("\n");
#print("First col in data : \n", arr1[:,-1]);
#
#traindata = pd.read_csv("training_set.csv");
#validationdata = pd.read_csv("validation_set.csv");

    print("\nThe Second decision tree:")
    fw.write('\nThe Second decision tree:\n')
    mytrees2=createTree(myDat,label2,1)

    if printTree == 'yes' :
        print()
        testprint(mytrees2,1)
        print()

    sum2 = len(myDat)
    correct2=0
    error2=0
    sum2,correct2,error2=test(mytrees2,label4,myDat,sum2,correct2,error2)
    print("Training：", (correct2 / sum2)*100, "%" )
    fw.write('Training：%.2f' % (correct2 / sum2))
    fw.write("\n")


    correct2=0
    error2=0
    sum2,correct2,error2=test(mytrees2,label4,myValidatDat,sum2,correct2,error2)
#print("\n")
    print("Validation：", (correct2 / sum2)*100, "%" )
    fw.write('Validation：%.2f' % (correct2 / sum2))
    fw.write("\n")

    correct2=0
    error2=0
    sum2,correct2,error2=test(mytrees2,label4,myTestDat,sum2,correct2,error2)
#print("\n")
    print("Test：", (correct2 / sum2)*100, "%" )
    fw.write('Test：%.2f' % (correct2 / sum2))
    fw.write("\n")



    print("\nThe Second decision tree with pruning:")
    fw.write('\nThe Second decision tree with pruning::\n')
    postPruning(mytrees2,myDat,myValidatDat,label5)

    if printPruneTree == 'yes' :
        print()
        testprint(mytrees2,1)
        print()

    correct2=0
    error2=0
    sum2,correct2,error2=test(mytrees2,label6,myDat,sum2,correct2,error2)
    print("Training：", (correct2 / sum2)*100, "%" )
    fw.write('Training：%.2f' % (correct2 / sum2))
    fw.write("\n")


    correct2=0
    error2=0
    sum2,correct2,error2=test(mytrees2,label6,myValidatDat,sum2,correct2,error2)
#print("\n")
    print("Validation：", (correct2 / sum2)*100, "%" )
    fw.write('Validation：%.2f' % (correct2 / sum2))
    fw.write("\n")

    correct2=0
    error2=0
    sum2,correct2,error2=test(mytrees2,label6,myTestDat,sum2,correct2,error2)
#print("\n")
    print("Test：", (correct2 / sum2)*100, "%" )
    fw.write('Test：%.2f' % (correct2 / sum2))
    fw.write("\n")
    fw.write("\n")
    fw.close()



main(sys.argv[1:]);

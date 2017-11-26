import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import tree
file = open("capture20110815-3.pcap.netflow.labeled","r")
X = []
Y = []
for line in file.readlines()[1:]:
    x=line.split()
    lis=[]
    try:
        for i in range(8,11):
            lis.append(int(x[i]))
        Y.append(str(x[len(x)-1]))
    except:
        print( lis)
    X.append(lis)
    #data.append(x[])
#print(len(data))

#data = data.reshape(-1,1)
#data = np.delete(data,5,axis=1)
##print(len(data[2]))
#data = np.asarray(data)

#print(Y)
X = np.asarray(X[1:])
Y = np.asarray(Y[1:])
print(X.shape)
print(Y.shape)
print (X)

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
print (Y)

le = preprocessing.LabelEncoder()
le.fit(Y)
print (le.classes_)
Y = le.transform(Y)

print (Y)
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(X, Y)

scaler = StandardScaler()
scaler.fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

clf = tree.DecisionTreeClassifier()

#clf = MLPClassifier(hidden_layer_sizes=(5,5,5,5),max_iter=500)

#print (X[2:4])
#X=np.asarray(X)
#Y=np.asarray(Y)
#print (X)
print(classification_report(y_train,y_train))

clf.fit(x_train, y_train)
print(x_test,y_test)
predictions = clf.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

y_labels = [0]*(4)
for i in predictions:
    if(i==0):
        y_labels[0]+=1
    elif(i==1):
        y_labels[1]+=1
    elif(i==2):
        y_labels[2]+=1
    else:
        y_labels[3]+=1
print (predictions)
print (y_labels)
x_labels=range(0,4)
## The line / model
plt.scatter(x_labels,y_labels)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

print ("Score:", clf.score(x_test, y_test))
#print(data[1][2])
#print(data)

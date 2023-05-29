
import numpy as np
import pandas as pd

benign=pd.read_csv('nbaiot-dataset/5.benign.csv')
g_c=pd.read_csv('nbaiot-dataset/5.gafgyt.combo.csv')
g_j=pd.read_csv('nbaiot-dataset/5.gafgyt.junk.csv')
g_s=pd.read_csv('nbaiot-dataset/5.gafgyt.scan.csv')
g_t=pd.read_csv('nbaiot-dataset/5.gafgyt.tcp.csv')
g_u=pd.read_csv('nbaiot-dataset/5.gafgyt.udp.csv')
m_a=pd.read_csv('nbaiot-dataset/5.mirai.ack.csv')
m_sc=pd.read_csv('nbaiot-dataset/5.mirai.scan.csv')
m_sy=pd.read_csv('nbaiot-dataset/5.mirai.syn.csv')
m_u=pd.read_csv('nbaiot-dataset/5.mirai.udp.csv')
m_u_p=pd.read_csv('nbaiot-dataset/5.mirai.udpplain.csv')

benign=benign.sample(frac=0.25,replace=False)
g_c=g_c.sample(frac=0.25,replace=False)
g_j=g_j.sample(frac=0.5,replace=False)
g_s=g_s.sample(frac=0.5,replace=False)
g_t=g_t.sample(frac=0.15,replace=False)
g_u=g_u.sample(frac=0.15,replace=False)
m_a=m_a.sample(frac=0.25,replace=False)
m_sc=m_sc.sample(frac=0.15,replace=False)
m_sy=m_sy.sample(frac=0.25,replace=False)
m_u=m_u.sample(frac=0.1,replace=False)
m_u_p=m_u_p.sample(frac=0.27,replace=False)

benign['type']='benign'
m_u['type']='mirai_udp'
g_c['type']='gafgyt_combo'
g_j['type']='gafgyt_junk'
g_s['type']='gafgyt_scan'
g_t['type']='gafgyt_tcp'
g_u['type']='gafgyt_udp'
m_a['type']='mirai_ack'
m_sc['type']='mirai_scan'
m_sy['type']='mirai_syn'
m_u_p['type']='mirai_udpplain'

data=pd.concat([benign,m_u,g_c,g_j,g_s,g_t,g_u,m_a,m_sc,m_sy,m_u_p],
               axis=0, sort=False, ignore_index=True)

sampler=np.random.permutation(len(data))
data=data.take(sampler)

labels_full=pd.get_dummies(data['type'], prefix='type')
target=np.array(data['type'].tolist())
labels=labels_full.values

data=data.drop(columns='type')

#applying standardization to the data
def standardize(df,col):
    df[col]= (df[col]-df[col].mean())/df[col].std()

data_st=data.copy()
for i in (data_st.iloc[:,:-1].columns):
    standardize (data_st,i)

train_data_st=data_st.values

print("Number of instances used-\n")
data.groupby('type')['type'].count()

from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# test/train split  25% test
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, labels, test_size=0.25, random_state=73)

#  create sequential model
model = Sequential()
model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(40, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(labels.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')
model.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
          callbacks=[monitor],verbose=2,epochs=500)


#predict using model
pred_st = model.predict(x_test_st)
pred_st = np.argmax(pred_st,axis=1)
y_eval_st = np.argmax(y_test_st,axis=1)
score_st = metrics.accuracy_score(y_eval_st, pred_st)


print("accuracy: {}".format(score_st))
print(classification_report(y_eval_st,pred_st))



from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    data.values, target, test_size=0.25, random_state=77)

#Create decision tree 
from sklearn.tree import DecisionTreeClassifier
dtree_model=DecisionTreeClassifier()
dtree_model = dtree_model.fit(x_train_st, y_train_st)
dtree_predictions = dtree_model.predict(x_test_st)
print("accuracy of decision tree- ",accuracy_score(y_test_st,dtree_predictions))
print(classification_report(y_test_st,dtree_predictions))  


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, target, test_size=0.25, random_state=49)

#create KNN model
knn = KNeighborsClassifier(n_neighbors=7)
  
knn.fit(x_train_st, y_train_st)
y_pred=knn.predict(x_test_st)

print("accuracy of KNN- ",accuracy_score(y_test_st,y_pred))
print(classification_report(y_test_st,y_pred)) 


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, target, test_size=0.25, random_state=43)

#training a SVM Classifier
svm_model_linear = SVC(kernel = 'poly', C = 30).fit(x_train_st, y_train_st)
svm_predictions = svm_model_linear.predict(x_test_st)


print("accuracy of SVM- ",accuracy_score(y_test_st,svm_predictions))
print(classification_report(y_test_st,svm_predictions))


from sklearn.model_selection import train_test_split


# dividing X, y into train and test data
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, target, test_size=0.25, random_state=55)


# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB().fit(x_train_st, y_train_st)
gnb_predictions = gnb.predict(x_test_st)

# accuracy on X_test
accuracy = gnb.score(x_test_st, y_test_st)
print ("accuracy of GNB- ",accuracy)
print(classification_report(y_test_st,gnb_predictions)) 


from sklearn.ensemble import RandomForestClassifier

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    data.values, target, test_size=0.25, random_state=67)

clf = RandomForestClassifier(n_estimators = 50, criterion="gini")

clf.fit(x_train_st, y_train_st)

# performing predictions on the test dataset
y_pred = clf.predict(x_test_st)

from sklearn import metrics

# using metrics module for accuracy calculation
print("accuracy of Random forest- ", metrics.accuracy_score(y_test_st, y_pred))
print(classification_report(y_test_st,y_pred))
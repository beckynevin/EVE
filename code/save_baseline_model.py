###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Creates a baseline model (no denoising) and saves it

# Runs a neural network by hand (not through the knodle framework)

# Requires knodle conda environment to run
# or (better yet) ciao_knodle environment

# Also, uses tensorboard and pytorch

# I'm experimenting with two things right now:
# 1) How to run the logistic regression 
#    (does it need a sigmoid or optimizer?)
# 2) Do I need to standardize the input data?

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
#import sys
#sys.path.append("..")
from utils_for_knodle import *
from my_knodle_models import LogisticRegression_no_float
from sklearn import preprocessing

# Create your own logistic regression in a loop please :)





NUM_OUTPUT_CLASSES = 1
model_type = 'logistic'
predictors = 'hyper_and_energy'
norm = True

# First step is to import your data
subsample = pd.read_csv('../data/mega_dfs/training.csv',sep='\t')
subsample_dev = pd.read_csv('../data/mega_dfs/dev.csv',sep='\t')
subsample_test = pd.read_csv('../data/mega_dfs/test.csv',sep='\t')




# No rules are necessary when running the baseline

feature_list_all = ['x', 'y', 'crsu', 'crsv', 'pha', 'sumamps', 'fp_u', 'fp_v', 'fb_u', 'fb_v']

if predictors == 'hyper':
    feature_list = ['fp_u', 'fp_v', 'fb_u', 'fb_v']#,'pha','sumamps','crsu','crsv']
if predictors == 'hyper_and_energy':
    feature_list = ['fp_u', 'fp_v', 'fb_u', 'fb_v','pha','sumamps']#,'crsu','crsv']


model_input_x_array = subsample[feature_list]#.to_numpy()

if norm:
    # Normalize the input x array:
    
    # l1, l2, max don't really make much of a difference
    # still getting super small #s when normalized
    # Used to be Normalizer
    normalizer = preprocessing.StandardScaler()#norm='l1')
    normalized_train_X = normalizer.fit_transform(model_input_x_array)
    

    model_input_x_array = normalizer.transform(model_input_x_array)

    

    

    model_input_x_dev = normalizer.transform(subsample_dev[feature_list])#.to_numpy()
    model_input_x_test = normalizer.transform(subsample_test[feature_list])
    model_input_x_array_all = subsample[feature_list]#.to_numpy()


else:
    model_input_x_array = model_input_x_array.to_numpy()
    model_input_x_array_all = subsample[feature_list].to_numpy()
    model_input_x_dev = subsample_dev[feature_list].to_numpy()
    model_input_x_test = subsample_test[feature_list].to_numpy()


# This is the non-normalized input (can later use it to run the model)
model_input_x_array_all = subsample[feature_list]#.to_numpy()

model_input_x = np_array_to_tensor(model_input_x_array).float()
model_input_x_dev = np_array_to_tensor(model_input_x_dev).float()
model_input_x_test = np_array_to_tensor(model_input_x_test).float()

# get the ys
y_train = np_array_to_tensor(subsample['class overall'].values).float()
y_train = y_train.view(y_train.shape[0],1)

y_test_hyper = np_array_to_tensor(subsample_test['class overall'].values).float()
y_test_hyper = y_test_hyper.view(y_test_hyper.shape[0],1)

y_test_steve = np_array_to_tensor(subsample_test['class overall'].values).float()
y_test_steve = y_test_steve.view(y_test_steve.shape[0],1)

y_test_stowed = np_array_to_tensor(subsample_test['class overall'].values).float()
y_test_stowed = y_test_stowed.view(y_test_stowed.shape[0],1)



#if __name__ == "__main__":
model = LogisticRegression_no_float(len(feature_list), NUM_OUTPUT_CLASSES)

learning_rate = 0.01
# Apparently its more numerically stable to use BCEWithLogitsLoss() 
# and then take out the sigmoid from the forward pass?
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

num_epochs = 10000
loss_list = []
accuracy_list = []
for epoch in range(num_epochs):
    y_predicted = torch.sigmoid(model(model_input_x))#X_train
    loss = criterion(y_predicted, y_train)
    loss_list.append(loss)

    #print('y_predicted', y_predicted)
    #print('y_train', y_train)
    acc = y_predicted.round().eq(y_train.round()).sum() / float(y_train.shape[0])
    #print(acc)
    
    #acc = np.mean(y_train == y_predicted)
    accuracy_list.append(acc)

    # backwards pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}, accuracy = {acc}')


#
plt.clf()
plt.plot(range(num_epochs), loss_list, label='loss')
plt.plot(range(num_epochs), accuracy_list, label='accuracy')
plt.legend()
plt.xlabel('epochs')
plt.show()

# Run on the test set:
y_predicted = torch.sigmoid(model(model_input_x_test))#X_train
print('output ys', y_predicted.detach().numpy())
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(y_predicted.detach().numpy())#.detach().numpy())
#ax.hist(ys[:,0].detach().numpy(), range=[-0.33,-0.32], bins=100)
ax.set_xlabel('ys')


plt.show()

    
acc = y_predicted.round().eq(y_test_hyper.round()).sum() / float(y_test_hyper.shape[0])
print('test accuracy hyper', acc)

acc = y_predicted.round().eq(y_test_steve).sum() / float(y_test_steve.shape[0])
print('test accuracy steve', acc)

acc = y_predicted.round().eq(y_test_stowed).sum() / float(y_test_stowed.shape[0])
print('test accuracy stowed', acc)

# Lets make confusion matrices

y_predicted_array = y_predicted.round().detach().numpy()
y_test_hyper_array = y_test_hyper.detach().numpy()
y_test_steve_array = y_test_steve.detach().numpy()
y_test_stowed_array = y_test_stowed.detach().numpy()

def calc_confusion(y_predicted_array, y_test_array):

    TP = np.sum([1 if (y_predicted_array[i] == 1 and y_test_array[i] == 1) else 0 for i, x in enumerate(y_test_array)])
    TN = np.sum([1 if (y_predicted_array[i] == 0 and y_test_array[i] == 0) else 0 for i, x in enumerate(y_test_array)])
    FP = np.sum([1 if (y_predicted_array[i] == 1 and y_test_array[i] == 0) else 0 for i, x in enumerate(y_test_array)])
    FN = np.sum([1 if (y_predicted_array[i] == 0 and y_test_array[i] == 1) else 0 for i, x in enumerate(y_test_array)])
    return TP, TN, FP, FN

TP, TN, FP, FN = calc_confusion(y_predicted_array, y_test_hyper_array)
print('hyper')
print('alternate accuracy', (TP + TN)/(TP+TN+FP+FN))
print('precision', TP / (TP + FP))
print('recall', TP / (TP + FN))

TP, TN, FP, FN = calc_confusion(y_predicted_array, y_test_steve_array)
print('steve')
print('alternate accuracy', (TP + TN)/(TP+TN+FP+FN))
print('precision', TP / (TP + FP))
print('recall', TP / (TP + FN))

TP, TN, FP, FN = calc_confusion(y_predicted_array, y_test_stowed_array)
print('stowed')
print('alternate accuracy', (TP + TN)/(TP+TN+FP+FN))
print('precision', TP / (TP + FP))
print('recall', TP / (TP + FN))



# Try to save the model 

# Options to save this:
model.feature_names = feature_list


model.data = model_input_x_array_all
model.ids = subsample['id']
if norm:
    model.normalizer = normalizer
else:
    model.normalizer = None
print('these are features of the model', model.__dict__)

# Also save the input ys
model.ys = subsample[['class hyper', 'class steve', 'class stowed','class overall']]

filename = '../models/logistic/baseline_'+str(predictors)+'.sav'
joblib.dump(model, filename)
print('saved model')

ys = torch.sigmoid(model.forward(model_input_x_test))
#print('after forward feed', ys)
#print(np.shape(ys))
#print(ys[:,0])

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(ys[:,0].detach().numpy())
#ax.hist(ys[:,0].detach().numpy(), range=[-0.33,-0.32], bins=100)
ax.set_xlabel('ys[:,0]')
ax1 = fig.add_subplot(122)
ax1.hist(y_train[:,0].detach().numpy())
#ax1.hist(ys[:,1].detach().numpy(), range=[-0.62,-0.61], bins=100)
ax1.set_xlabel('ys training')
plt.show()



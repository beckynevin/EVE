###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Creates knodle models and saves them

# Requires knodle conda environment to run
# or (better yet) ciao_knodle environment

# Also, uses tensorboard

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from knodle.model.logistic_regression_model import LogisticRegressionModel

import random
import pandas as pd
import numpy as np
import joblib
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from transformers import AdamW

import matplotlib.pyplot as plt

# Import all trainers and configs:
from knodle.trainer.multi_trainer import MultiTrainer
from knodle.trainer.trainer import BaseTrainer
from knodle.trainer.knn_aggregation.knn import KNNAggregationTrainer
from knodle.trainer.baseline.majority import MajorityVoteTrainer

from knodle.trainer import AutoTrainer, AutoConfig, TrainerConfig
from knodle.trainer import MajorityConfig, KNNConfig, SnorkelConfig, SnorkelKNNConfig


NUM_OUTPUT_CLASSES = 2
trainer_type = 'majority'
#'base'
norm = False
model_type = 'logistic'
predictors = 'hyper_and_energy'

# First step is to import your data
# This has 1e7 lines of data:
df = pd.read_csv('../data/mega_dfs/weak_supervision_with_steve.csv', sep='\t')
#df = df[df['id']=='1505']

df = df[(df['id']=='1505') | (df['id']=='1287') | (df['id']=='579') | (df['id']=='hrciD2007-01-01bkgrndN0002.fits')]

print('original length of weak_supervision_with_steve.csv', len(df))
print(df['class stowed'])
df['class stowed'] = df['class stowed'].fillna(0)

df = df.dropna()
print('length after dropping nans', len(df))
print(df['class stowed'])

print(df.columns)







# also create a dev sample:

# Get a smaller sample please :0
smaller_sample = df.sample(n=int(1e5))

print(smaller_sample['id'].value_counts())


# Normalize this

subsample = smaller_sample.sample(frac = 0.8)
print('len of training', len(subsample))

rest_subsample = smaller_sample.drop(subsample.index)
# Now split this 50/50 into an CV and a test sample:

subsample_dev = rest_subsample.sample(frac = 0.5)
subsample_test = rest_subsample.drop(subsample_dev.index)

print('len of CV', len(subsample_dev))
print('len of test', len(subsample_test))



print(subsample.columns)

# So the input to knodle is:
# model_input_x: Your model features without any labels. Shape: (n_instances x features)
# mapping_rules_labels_t: This matrix maps all weak rules to a label. Shape: (n_rules x n_classes)
#   We will have 2 rules here, I assume its hot encoding, and 2 classes
#   In our case everything activates background
#   [[0, 1], [0, 1]]
# rule_matches_z: This matrix shows all applied rules on your dataset. Shape: (n_instances x n_rules)

rules = subsample[['class hyper','class steve','class stowed']]
# Modify the hyperscreen rule to throw a 1 when background
# so this means everywhere that it throws a 0.5 should be a 1 and -0.5 should be a 0
print(rules)
'''
rules['class hyper'] = rules['class hyper'].replace([-0.5],int(0))
rules['class hyper'] = rules['class hyper'].replace([0.5],int(1))

rules['class steve'] = rules['class steve'].replace([-0.5],int(0))
rules['class steve'] = rules['class steve'].replace([0.5],int(1))

# A negative 1 means that it is abstaining
rules['class stowed'] = rules['class stowed'].replace([0],int(-1))
'''

rule_matches_z = rules.to_numpy()
print(rules)
print(rules.value_counts())
print(rules['class hyper'].value_counts())
print(rules['class steve'].value_counts())
print(rules['class stowed'].value_counts())

print(rule_matches_z)
print('shape of rules_matches_z', np.shape(rule_matches_z))




in_rules = rule_matches_z




mapping_rules_labels_t = np.array([[0.,1.],[0.,1.],[1.,0.]])
print('shape of mapping rules to labels', np.shape(mapping_rules_labels_t))

# Try actually multiplying these together
print('Z = ', rule_matches_z)
print('T = ', mapping_rules_labels_t)
print('Y = ZT', np.dot(rule_matches_z,mapping_rules_labels_t))




feature_list_all = ['x', 'y', 'crsu', 'crsv', 'pha', 'sumamps', 'fp_u', 'fp_v', 'fb_u', 'fb_v']

if predictors == 'hyper':
    feature_list = ['fp_u', 'fp_v', 'fb_u', 'fb_v']#,'pha','sumamps','crsu','crsv']
if predictors == 'hyper_and_energy':
    feature_list = ['fp_u', 'fp_v', 'fb_u', 'fb_v','pha','sumamps']#,'crsu','crsv']


model_input_x_array = subsample[feature_list]#.to_numpy()

if norm:
    # Normalize the input x array:
    from sklearn import preprocessing
    # l1, l2, max don't really make much of a difference
    # still getting super small #s when normalized
    normalizer = preprocessing.Normalizer(norm='l1')
    normalized_train_X = normalizer.fit_transform(model_input_x_array)
    print('normalized X', normalized_train_X)
    print('normalizer', normalizer)
    print('elements of normalizer', normalizer.__dict__)

    print('before normalized', model_input_x_array)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.hist(model_input_x_array['fp_u'].values, color='#EC0868')
    ax.set_xlabel('fp_u pre norm')

    model_input_x_array = normalizer.transform(model_input_x_array)

    ax1 = fig.add_subplot(122)
    ax1.hist(model_input_x_array[:,0], color='#EC0868')
    ax1.set_xlabel('after norm')
    plt.show()

    print('after normalized', model_input_x_array, np.shape(model_input_x_array))

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

model_input_x = np_array_to_tensor_dataset(model_input_x_array)
model_input_x_dev = np_array_to_tensor_dataset(model_input_x_dev)
model_input_x_test = np_array_to_tensor_dataset(model_input_x_test)
#mapping_rules_labels_t = np_array_to_tensor_dataset(mapping_rules_labels_t)
#rule_matches_z = np_array_to_tensor_dataset(rule_matches_z)

#subsample_dev['class hyper'] = subsample_dev['class hyper'].replace([-0.5],int(0))
#subsample_dev['class hyper'] = subsample_dev['class hyper'].replace([0.5],int(1))
#subsample_dev['class steve'] = subsample_dev['class steve'].replace([-0.5],int(0))
#subsample_dev['class steve'] = subsample_dev['class steve'].replace([0.5],int(1))


#y_dev = np_array_to_tensor_dataset(subsample_dev['class hyper'].values)




# The predictive model is going to be contained within 
# the trainer class
logreg_model = LogisticRegressionModel(model_input_x_array.shape[1], 2)

learning_rate = 0.0001
# default criterion looks to be: 
# from snorkel.classification import cross_entropy_with_probs

# default loss is SGD

#optimizer=SGD(logreg_model.parameters(), lr = learning_rate)
# I think this is for when you have a 1D output:
# criterion = torch.nn.BCEWithLogitsLoss()
configs = MajorityConfig( optimizer = AdamW, lr = learning_rate, batch_size=32, epochs=20)#, filter_non_labeled = False)
    #optimizer=AdamW, lr=1e-4, batch_size=16, epochs=3)]#,
#    KNNConfig(optimizer=AdamW, k=2, lr=1e-4, batch_size=32, epochs=2),
#    SnorkelConfig(optimizer=AdamW),
#    SnorkelKNNConfig(optimizer=AdamW, radius=0.8),
#    WSCrossWeighConfig(optimizer=AdamW)
#]



trainer = MajorityVoteTrainer(
    #name=["majority"],#, "knn", "snorkel", "snorkel_knn", "wscrossweigh"],
    model=logreg_model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=model_input_x,
    rule_matches_z=rule_matches_z,
    trainer_config=configs, 
)
'''

configs = KNNConfig(optimizer=AdamW, k=10, lr=1e-4, batch_size=32, epochs=5)


trainer = KNNAggregationTrainer(
    model=logreg_model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=model_input_x,
    rule_matches_z=rule_matches_z,
    trainer_config=configs, 
)
'''



trainer.input_rules_matches_z = in_rules

random_index_list = random.sample(range(np.shape(model_input_x_array)[0]), 30)



make_rules_and_y_plot(in_rules, mapping_rules_labels_t, random_index_list)

# What we ultimately want to do is train and then save the model itself


trainer.train()
print('model has been trained')

STOP
try:
    out_rules = trainer._knn_denoise_rule_matches()
except AttributeError:
    out_rules = trainer.rule_matches_z_majority_vote#.toarray()

if np.shape(in_rules)[0] != np.shape(out_rules)[0]:
    print('shapes are not the same', np.shape(in_rules)[0], np.shape(out_rules)[0])
    # then this means that we are going to need to pad a new out_rules array
    # because the rols that had all zeros were dropped by the trainer

    
    count_zeros = 0
    index_dropped_1 = []

    for i in range(np.shape(in_rules)[0]):
        if np.all((in_rules[i] == 0)):
            index_dropped_1.append(i)
            count_zeros+=1


    out_rules_padded = out_rules 
    out_y_padded = trainer.noisy_y_train
    # but now add back in rows of zero
    for row in index_dropped_1:
        out_rules_padded = np.insert(out_rules_padded, row, [0, 0, 0], axis=0)
        out_y_padded = np.insert(out_y_padded, row, [0, 0], axis=0)
    
    # Now check if its padded in the same places:
    index_dropped_2 = []

    for i in range(np.shape(in_rules)[0]):
        if np.all((out_rules_padded[i] == 0)):
            index_dropped_2.append(i)


    if all(index_dropped_1[x] == index_dropped_2[x] for x in range(len(index_dropped_1))):
        print('all zeros are same index')
    out_rules = out_rules_padded
    out_y = out_y_padded

else:
    print('shapes are somehow the same', np.shape(in_rules)[0], np.shape(out_rules)[0])
    if np.shape(in_rules)[0] != np.shape(trainer.noisy_y_train)[0]:
        print('y is not the same as input', np.shape(trainer.noisy_y_train))
        count_zeros = 0
        index_dropped_1 = []

        for i in range(np.shape(in_rules)[0]):
            if np.all((in_rules[i] == 0)):
                index_dropped_1.append(i)
                count_zeros+=1


        out_y_padded = trainer.noisy_y_train
        # but now add back in rows of zero
        for row in index_dropped_1:
            out_y_padded = np.insert(out_y_padded, row, [0, 0], axis=0)
        
        
        out_y = out_y_padded
    else:
        out_y = trainer.noisy_y_train

trainer.output_rules_matches_z = out_rules
trainer.noisy_y = out_y



print('output y', out_y)

tensor_input = torch.from_numpy(model_input_x_array_all[feature_list].to_numpy())

ys = trainer.model.forward(tensor_input)
print('after forward feed', ys)
print(np.shape(ys))
print(ys[:,0])

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(ys[:,0].detach().numpy())
#ax.hist(ys[:,0].detach().numpy(), range=[-0.33,-0.32], bins=100)
ax.set_xlabel('ys[:,0]')
ax1 = fig.add_subplot(122)
ax1.hist(ys[:,1].detach().numpy())
#ax1.hist(ys[:,1].detach().numpy(), range=[-0.62,-0.61], bins=100)
ax1.set_xlabel('ys[:,1]')
plt.show()
STOP




make_rules_and_y_plot(out_rules, mapping_rules_labels_t, random_index_list)
make_rules_and_double_y_plot(out_rules, mapping_rules_labels_t,out_y, random_index_list)
print('output rules shape', np.shape(out_rules))



# Options to save this:
trainer.feature_names = feature_list
print('dictionary', trainer.__dict__)


trainer.data = model_input_x_array_all
trainer.ids = subsample['id']
if norm:
    trainer.normalizer = normalizer
else:
    trainer.normalizer = None


res = np.flatnonzero(in_rules != out_rules)
print('trying to find differences', res)
try:
    print('length of changes', len(res))
except:
    print('no length')

res2 = np.where((in_rules-out_rules) != 0)
print('diff 2', res2)


df_out_rules = pd.DataFrame(out_rules, columns = ['class hyper', 'class steve', 'class stowed'])
print(df_out_rules)
print(df_out_rules['class hyper'].value_counts())
print(df_out_rules['class steve'].value_counts())
print(df_out_rules['class stowed'].value_counts())







filename = '../models/logistic/majority_'+str(predictors)+'.sav'
joblib.dump(trainer, filename)
print('saved model')

# Okay I think I'd now like to incorporate some sick interpretation into here
# Two ways: 
# 1) testing on the three different labels as 'truth' in the test set
# 2) getting more than just accuracy from these labels

# subsample_test
y_test_hyper = np_array_to_tensor_dataset(subsample_test['class hyper'].values)
y_test_steve = np_array_to_tensor_dataset(subsample_test['class steve'].values)
y_test_stowed = np_array_to_tensor_dataset(subsample_test['class stowed'].values)

# Run evaluation
#eval_dict, _ = trainer.test(X_test, y_test)
metric, _ = trainer.test(model_input_x_test, y_test_hyper)
print('metric', metric)
print(f"Hyper test: accuracy: {metric.get('accuracy')}")

metric, _ = trainer.test(model_input_x_test, y_test_steve)
print(f"Steve test: accuracy: {metric.get('accuracy')}, loss: {metric.get('loss')}")

metric, _ = trainer.test(model_input_x_test, y_test_stowed)
print(f"Stowed test: accuracy: {metric.get('accuracy')}, precision: {metric.get('precision')}")



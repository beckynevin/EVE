###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# created 7/26/22

# Creates knodle models and saves them

# This one focuses on knn

# Requires knodle conda environment to run
# or (better yet) ciao_knodle environment

# Also, uses tensorboard

# This is the version that ones one dimensional input instead of two 
# for logistic regression

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import configparser
from knodle.model.logistic_regression_model import LogisticRegressionModel

import random
import pandas as pd
import numpy as np
import joblib
import torch


from my_knodle_models import TinyModel, MassiveModel, SigmoidModel

from torch.utils.tensorboard import SummaryWriter


#from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
#from utils.utils_ML import np_array_to_tensor_dataset
from torch.optim import SGD
from torch.optim import Adam
import sys
sys.path.append("..")
from transformers import AdamW

import matplotlib.pyplot as plt

# Import all trainers and configs:

from knodle.trainer.wscrossweigh.wscrossweigh import WSCrossWeighTrainer

from knodle.trainer.wscrossweigh.config import WSCrossWeighConfig
from utils_for_knodle import *



NUM_OUTPUT_CLASSES = 2
trainer_type = 'majority'
#'base'
norm = True
model_type = 'logistic'
predictors = 'hyper_and_energy'

# First step is to import your data
subsample = pd.read_csv('../data/mega_dfs/training.csv',sep='\t')
subsample_dev = pd.read_csv('../data/mega_dfs/dev.csv',sep='\t')
subsample_test = pd.read_csv('../data/mega_dfs/test.csv',sep='\t')

print('shape of training', len(subsample))
print('shape of test', len(subsample_test))



# gotta switch the 0s and 1s in all of the stowed columns 
subsample['class stowed'] = subsample['class stowed'].replace(0,100)
subsample['class stowed'] = subsample['class stowed'].replace(1,0)
subsample['class stowed'] = subsample['class stowed'].replace(100,1)

subsample_dev['class stowed'] = subsample_dev['class stowed'].replace(0,100)
subsample_dev['class stowed'] = subsample_dev['class stowed'].replace(1,0)
subsample_dev['class stowed'] = subsample_dev['class stowed'].replace(100,1)

subsample_test['class stowed'] = subsample_test['class stowed'].replace(0,100)
subsample_test['class stowed'] = subsample_test['class stowed'].replace(1,0)
subsample_test['class stowed'] = subsample_test['class stowed'].replace(100,1)



# I modified these to all throw ones when foreground
rules = subsample[['class hyper','class steve','class stowed']]

# subsample is training rules
rule_matches_z = rules.to_numpy()

in_rules = rule_matches_z




mapping_rules_labels_t = np.array([[0., 1.],[0., 1.],[2., 0.]])
print('shape of mapping rules to labels', np.shape(mapping_rules_labels_t))

# Try actually multiplying these together
print('Z = ', rule_matches_z, np.shape(rule_matches_z))
print('T = ', mapping_rules_labels_t, np.shape(mapping_rules_labels_t))
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
    # for normalizer():
    # l1, l2, max don't really make much of a difference
    # still getting super small #s when normalized
    normalizer = preprocessing.StandardScaler()#norm='l1')
    normalized_train_X = normalizer.fit_transform(model_input_x_array)
    
    '''
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.hist(model_input_x_array['pha'].values, color='#EC0868')
    ax.set_xlabel('pha pre norm')

    model_input_x_array = normalizer.transform(model_input_x_array)

    ax1 = fig.add_subplot(122)
    ax1.hist(model_input_x_array[:,0], color='#EC0868')
    ax1.set_xlabel('after norm')
    plt.show()

    print('after normalized', model_input_x_array, np.shape(model_input_x_array))
    '''

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

model_input_x_tensorset = np_array_to_tensor_dataset(model_input_x_array)
model_input_x_dev_tensorset = np_array_to_tensor_dataset(model_input_x_dev)
model_input_x_test_tensorset = np_array_to_tensor_dataset(model_input_x_test)

model_input_x_tensor = np_array_to_tensor(model_input_x_array).float()
model_input_x_dev_tensor = np_array_to_tensor(model_input_x_dev).float()
model_input_x_test_tensor = np_array_to_tensor(model_input_x_test).float()


# The predictive model is going to be contained within 
# the trainer class
logreg_model = LogisticRegressionModel(len(feature_list), NUM_OUTPUT_CLASSES)




# default criterion looks to be: 
# from snorkel.classification import cross_entropy_with_probs

# default loss is SGD

#optimizer=SGD(logreg_model.parameters(), lr = learning_rate)
#torch.nn.BCEWithLogitsLoss
parameters = {
        "lr": 1e-4, "cw_lr": 0.8, "epochs": 5, "cw_partitions": 2, "cw_folds": 5, "cw_epochs": 2, "weight_rr": 0.7,
        "samples_start_weights": 4.0
    }
# to have sample weights saved with some specific index in the file name, you can use "caching_suffix" variable
caching_suffix = f"dscw_{parameters.get('cw_partitions')}part_{parameters.get('cw_folds')}folds_" \
                     f"{parameters.get('weight_rr')}wrr"

configs = WSCrossWeighConfig(
        # general trainer parameters
        output_classes=NUM_OUTPUT_CLASSES,
        filter_non_labelled=False,
        #other_class_id=3,
        seed=12345,
        epochs=parameters.get("epochs"),
        batch_size=16,
        optimizer=AdamW,
        lr=parameters.get("lr"),
        grad_clipping=5,
        caching_suffix=caching_suffix,
        #saved_models_dir=os.path.join(path_to_data, "trained_models"),  # trained classifier model will be saved after each epoch

        # WSCrossWeigh specific parameters
        partitions=parameters.get("cw_partitions"),  # number of WSCrossWeigh iterations (= splitting into folds)
        folds=parameters.get("cw_folds"),  # number of folds train data will be splitted into
        weight_reducing_rate=parameters.get("weight_rr"),  # sample weights reducing coefficient
        samples_start_weights=parameters.get("samples_start_weights"),  # the start weight of sample weights
        cw_epochs=parameters.get("cw_epochs"),  # number of epochs each WSCrossWeigh model is to be trained
        cw_optimizer=Adam,  # WSCrossWeigh model optimiser
        cw_lr=parameters.get("cw_lr")  # WSCrossWeigh model lr
    )
#KNNConfig(optimizer=AdamW, k=5, lr=1e-3, batch_size=32, epochs=20)


trainer = WSCrossWeighTrainer(
        # general Trainer inputs (a more detailed explanation of Knodle inputs is in README)
        model=logreg_model,  # classification model
        mapping_rules_labels_t=mapping_rules_labels_t,  # t matrix
        model_input_x=model_input_x_tensorset,  # x matrix for training the classifier
        rule_matches_z=rule_matches_z,  # z matrix
        trainer_config=configparser)#,

        # additional dev set used for classification model evaluation during training
        #dev_model_input_x=model_input_x_dev_tensorset,
        #dev_gold_labels_y=y_dev,

        # WSCrossWeigh specific parameters. If they are not defined, the corresponding main classification parameters
        # will be used instead (model instead of cw_model etc)
        #cw_model=cw_model,  # model that will be used for WSCrossWeigh weights calculation
        #cw_model_input_x=train_dataset_tfidf,  # x matrix for training the WSCrossWeigh models
    #)


'''
trainer = KNNAggregationTrainer(
    model=logreg_model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=model_input_x_tensorset,
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






out_rules = trainer._knn_denoise_rule_matches()

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
        out_y_padded = np.insert(out_y_padded, row, [0], axis=0)
    
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
            out_y_padded = np.insert(out_y_padded, row, [0], axis=0)
        
        
        out_y = out_y_padded
    else:
        out_y = trainer.noisy_y_train

trainer.output_rules_matches_z = out_rules
trainer.noisy_y = out_y



print('output y', out_y)

#tensor_input = torch.from_numpy(model_input_x_dev.numpy())
#torch.from_numpy(model_input_x_array_all[feature_list].to_numpy())

print('okay feeding the model this', model_input_x_tensor)
#print('alternate way of seeing this', tensor_input.detach().numpy())

ys = torch.sigmoid(trainer.model(model_input_x_tensor)).detach().numpy()
print('after forward feed', ys)




#make_rules_and_y_plot(out_rules, mapping_rules_labels_t, random_index_list)
#make_rules_and_double_y_plot(out_rules, mapping_rules_labels_t, ys, random_index_list)
make_double_rules_and_double_y_plot(in_rules, out_rules, mapping_rules_labels_t, ys, random_index_list)
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
metric, _ = trainer.test(model_input_x_test_tensorset, y_test_hyper)
print('metric', metric)
print(f"Hyper test: accuracy: {metric.get('accuracy')}")

metric, _ = trainer.test(model_input_x_test_tensorset, y_test_steve)
print(f"Steve test: accuracy: {metric.get('accuracy')}, loss: {metric.get('loss')}")

metric, _ = trainer.test(model_input_x_test_tensorset, y_test_stowed)
print(f"Stowed test: accuracy: {metric.get('accuracy')}, precision: {metric.get('precision')}")


y_test_hyper = np_array_to_tensor(subsample_test['class hyper'].values).float()
y_test_hyper = y_test_hyper.view(y_test_hyper.shape[0],1)

y_test_steve = np_array_to_tensor(subsample_test['class steve'].values).float()
y_test_steve = y_test_steve.view(y_test_steve.shape[0],1)

y_test_stowed = np_array_to_tensor(subsample_test['class stowed'].values).float()
y_test_stowed = y_test_stowed.view(y_test_stowed.shape[0],1)


filename = '../models/logistic/wscrossweigh_'+str(predictors)+'.sav'
joblib.dump(trainer, filename)
print('saved model')

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Creates knodle models and saves them

# Requires knodle conda environment to run

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from minio import Minio
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.baseline.majority import MajorityVoteTrainer
import pandas as pd
import numpy as np
import joblib
import torch
from torch.utils.data import TensorDataset
import scipy.sparse as sp
from my_knodle_models import TinyModel
from knodle.trainer import AutoTrainer, AutoConfig
from knodle.trainer.knn_aggregation.knn import KNNAggregationTrainer
#from utils.utils_ML import np_array_to_tensor_dataset

# Took this from the knodle github for converting between arrays and tensors
def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    """
    :rtype: object
    """
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x

NUM_OUTPUT_CLASSES = 2

# First step is to import your data
# This has 1e7 lines of data:
df = pd.read_csv('../data/mega_dfs/weak_supervision_with_steve.csv', sep='\t')
print('original length of weak_supervision_with_steve.csv', len(df))

df = df.dropna()
print('length after dropping nans', len(df))

print(df.columns)



# also create a dev sample:

# Get a smaller sample please :0
smaller_sample = df.sample(n=int(1e5))

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

rules = subsample[['class stowed','class hyper','class steve']]
# Modify the hyperscreen rule to throw a 1 when background
# so this means everywhere that it throws a 0.5 should be a 1 and -0.5 should be a 0
print(rules)
rules['class hyper'] = rules['class hyper'].replace([-0.5],int(0))
rules['class hyper'] = rules['class hyper'].replace([0.5],int(1))

rules['class steve'] = rules['class steve'].replace([-0.5],int(0))
rules['class steve'] = rules['class steve'].replace([0.5],int(1))
rule_matches_z = rules.to_numpy()
print(rules)
print(rule_matches_z)
print('shape of rules_matches_z', np.shape(rule_matches_z))



mapping_rules_labels_t = np.array([[0.,1.],[0.,1.],[0.,1.]])
print('shape of mapping rules to labels', np.shape(mapping_rules_labels_t))

# Try actually multiplying these together
print('Z = ', rule_matches_z)
print('T = ', mapping_rules_labels_t)
print('Y = ZT', np.dot(rule_matches_z,mapping_rules_labels_t))


feature_list_all = ['x', 'y', 'crsu', 'crsv', 'pha', 'sumamps', 'fp_u', 'fp_v', 'fb_u', 'fb_v']
feature_list = ['pha','sumamps','fp_u', 'fp_v', 'fb_u', 'fb_v','crsu','crsv']


model_input_x_array = subsample[feature_list]#.to_numpy()
model_input_x_array_all = subsample[feature_list_all]#.to_numpy()
model_input_x_dev = subsample_dev[feature_list]#.to_numpy()

'''
'x', 'y', 'crsu', 'crsv', 'pha', 'pi', 'sumamps', 'au1', 'au2', 'au3', 'av1', 'av2', 'av3'
'''

print('shape of model_input_x', np.shape(model_input_x_array))
print(model_input_x_array)








print('is this the right rules matching Z?', rule_matches_z)
print('shape', np.shape(rule_matches_z))
print(type(rule_matches_z))



print('is this the right mappings to rules T?', mapping_rules_labels_t)
print('shape', np.shape(mapping_rules_labels_t))
print(type(mapping_rules_labels_t))

model_input_x = np_array_to_tensor_dataset(model_input_x_array.values)
model_input_x_dev = np_array_to_tensor_dataset(model_input_x_dev.values)
#mapping_rules_labels_t = np_array_to_tensor_dataset(mapping_rules_labels_t)
#rule_matches_z = np_array_to_tensor_dataset(rule_matches_z)

subsample_dev['class hyper'] = subsample_dev['class hyper'].replace([-0.5],int(0))
subsample_dev['class hyper'] = subsample_dev['class hyper'].replace([0.5],int(1))
subsample_dev['class steve'] = subsample_dev['class steve'].replace([-0.5],int(0))
subsample_dev['class steve'] = subsample_dev['class steve'].replace([0.5],int(1))

'''
y_dev_list = np_array_to_tensor_dataset(subsample_dev['class hyper'].values)
y_dev = np.zeros((len(y_dev_list), int(max(y_dev_list)+1)))
y_dev[np.arange(len(y_dev_list)),y_dev_list] = 1
print('list', y_dev_list)
print('one hot', y_dev)
#y_dev = torch.nn.functional.one_hot(y_dev_list)

print(torch.nn.functional.one_hot(subsample_dev['class hyper'].values))
'''
y_dev = np_array_to_tensor_dataset(subsample_dev['class hyper'].values)




# The predictive model is going to be contained within 
# the trainer class
model = LogisticRegressionModel(
  model_input_x_array.shape[1], NUM_OUTPUT_CLASSES)

# I am needing to make me own models
# Knodle is automatically set up to enable a cross-entropy loss, which includes a softmax
'''
from snorkel.classification import cross_entropy_with_probs
class TrainerConfig:
    def __init__(
            self,
            criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
            batch_size: int = 32,
            optimizer: Optimizer = None,
            lr: int = 0.01,
            output_classes: int = 2,
            class_weights: Tensor = None,
            epochs: int = 3,
            seed: int = None,
            grad_clipping: int = None,
            device: str = None,
            caching_folder: str = os.path.join(pathlib.Path().absolute(), "cache"),
            caching_suffix: str = "",
            saved_models_dir: str = None,
            multi_label: bool = False,
            multi_label_threshold: float = None
    ):
        """
        A default and minimum sufficient configuration of a Trainer instance.
        :param criterion: a usual PyTorch criterion; computes a gradient according to a given loss function
        :param batch_size: a usual PyTorch batch_size; the number of training examples utilized in one training iteration
        :param optimizer: a usual PyTorch optimizer; which is used to solve optimization problems by minimizing the
        function
        :param lr: a usual PyTorch learning rate; tuning parameter in an optimization algorithm that determines the step
        size at each iteration while moving toward a minimum of a loss function
        :param output_classes: the number of classes used in classification
        :param class_weights: introduce the weight of each class. By default, all classes have the same weights 1.0.
        :param epochs: the number of epochs the classification model will be trained
        :param seed: the desired seed for generating random numbers
        :param grad_clipping: if set to True, gradient norm of an iterable of parameters will be clipped
        :param device: what device the model will be trained on (CPU/CUDA)
        :param caching_folder: a path to the folder where cache will be saved (default: root/cache)
        :param caching_suffix: a specific index that could be added to the caching file name (e.g. in WSCrossWeigh for
        sample weights calculated in different iterations and stored in different files.)
        :param saved_models_dir: a path to the folder where trained models will be stored. If None, the trained models
        :param multi_label: a boolean value whether the classification is multi-label
        won't be stored.
        """
'''
model = TinyModel(
    model_input_x_array.shape[1], NUM_OUTPUT_CLASSES)

'''
print('made model', model)
print('shape of y_dev', y_dev.shape)


# What are the autos for knodle?
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate (train_loader):
        images = images.reshape(-1, 28*28).to(device) # pushing to GPU
        labels = labels.to(device)
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0 :
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# testing and eval
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100. * n_correct / n_samples
'''

# rules_matching_z is n x 3
# mapping_rules_labels_T is 3 x 2 
# Y = ZT --> giving it dimensions of n x 2, gonna need to one hot encode the y_dev
# so y_dev should also be 

#print(subsample_dev['class hyper'].values)
#print(mapping_rules_labels_t*rule_matches_z)
#print(rule_matches_z*mapping_rules_labels_t)



trainer_type = "majority"
trainer_type = 'knn'
custom_model_config = AutoConfig.create_config(name = trainer_type, filter_non_labelled=False)

print(custom_model_config)

#kNNAggregationTrainer

# I'm concerned that the majority voter doesn't work for when there's only two rules
# Trying to add Steve's classification here to have 3 rules total



trainer = KNNAggregationTrainer(
  model=model,
  mapping_rules_labels_t=mapping_rules_labels_t,
  model_input_x=model_input_x,
  rule_matches_z=rule_matches_z, trainer_config=custom_model_config
)

'''
,
  dev_model_input_x=model_input_x_dev,
  dev_gold_labels_y=y_dev
'''



# So, knodle is weird because what you adjust is the trainer,
# which contains the model inside of it already

trainer.input_rules_matches_z = rule_matches_z



# What we ultimately want to do is train and then save the model itself
trainer.train()

print('noisy rules', rule_matches_z)
print('denoising the rules', trainer._knn_denoise_rule_matches())

print('shape noisy rules', np.shape(rule_matches_z))
print('shape denoising the rules', np.shape(trainer._knn_denoise_rule_matches()))

print('saving model + trainer ')
# Options to save this:
trainer.feature_names = feature_list
trainer.output_rules_matches_z = trainer._knn_denoise_rule_matches()
trainer.data = model_input_x_array_all
trainer.ids = subsample['id']


filename = '../models/kNN_hyper_and_energy_and_crs_NN.sav'
joblib.dump(trainer, filename)

print('saved model')

# the trained model is tested on the test set
clf_report, _ = trainer.test(X_test, y_test)
print(clf_report)

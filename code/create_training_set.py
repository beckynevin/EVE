###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reads in the mega file and saves a training set
# after modifying all of the labels 
# and adding a majority column 

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
#sys.path.append("..")
#from utils.utils_for_knodle import *


# Import the mega table which has 1e7 lines of code:
df = pd.read_csv('../data/mega_dfs/weak_supervision_with_steve.csv', sep='\t')
#df = df[df['id']=='1505']

# Optional way to select from specific IDs
# I think I added this because there were a lot more backgrounds than anything else
#df = df[(df['id']=='1505') | (df['id']=='1287') | (df['id']=='579') | (df['id']=='hrciD2007-01-01bkgrndN0002.fits')]

# fill everything that isn't background wtih zeros
df['class stowed'] = df['class stowed'].fillna(0)

df = df.dropna()


# Get a smaller sample please :0
# This randomly selects
smaller_sample = df.sample(n=int(1e5))

# Take this smaller sample and reassign the rules for the stowed class:
# Gotta swap everything so if there's a zero it becomes 1, meaning real
# -1 (which was actually labeled) becomes a zero, or background
smaller_sample['class stowed'] = smaller_sample['class stowed'].replace([1],int(-1))
smaller_sample['class stowed'] = smaller_sample['class stowed'].replace([0],int(1))
smaller_sample['class stowed'] = smaller_sample['class stowed'].replace([-1],int(0))

# Make an overall classification that is the sum of 
# class stowed, class hyper, class steve
smaller_sample['class overall'] = 1/3. * smaller_sample['class hyper'] + 1/3. * smaller_sample['class steve'] + 1/3. * smaller_sample['class stowed']

print(smaller_sample[['class hyper', 'class steve', 'class stowed','class overall']])

# Now, from the smaller sample, take 80%, this will be the training set
subsample = smaller_sample.sample(frac = 0.8)

# Now get everything else:
rest_subsample = smaller_sample.drop(subsample.index)
# Now split this 50/50 into an CV and a test sample:
subsample_dev = rest_subsample.sample(frac = 0.5)
subsample_test = rest_subsample.drop(subsample_dev.index)

# Now save these 
subsample.to_csv('../data/mega_dfs/training_2.csv',sep='\t')
subsample_dev.to_csv('../data/mega_dfs/dev_2.csv',sep='\t')
subsample_test.to_csv('../data/mega_dfs/test_2.csv',sep='\t')

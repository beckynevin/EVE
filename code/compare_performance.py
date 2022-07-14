# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compare the performance of various models

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.ma as ma
# import seaborn as sns
from scipy.stats.contingency import margins
import pandas as pd
# import astropy.io.fits as fits
import sys
sys.path.append("..")
from utils_for_knodle import np_array_to_tensor

def calc_confusion(y_predicted_array, y_test_array):
    #print('tp list', [1 if (y_predicted_array[i] == 1 and y_test_array[i] == 1) else 0 for i, x in enumerate(y_test_array)])
    TP = np.sum([1 if (y_predicted_array[i] == 1 and y_test_array[i] == 1) else 0 for i, x in enumerate(y_test_array)])
    TN = np.sum([1 if (y_predicted_array[i] == 0 and y_test_array[i] == 0) else 0 for i, x in enumerate(y_test_array)])
    FP = np.sum([1 if (y_predicted_array[i] == 1 and y_test_array[i] == 0) else 0 for i, x in enumerate(y_test_array)])
    FN = np.sum([1 if (y_predicted_array[i] == 0 and y_test_array[i] == 1) else 0 for i, x in enumerate(y_test_array)])
    return TP, TN, FP, FN



# So there are two options here:
# 1) Pull from a new evt1 file (evt1 = True)
# 2) Pull from a saved csv file (massive_file = True)

plot = False
evt1 = False  # True
massive_file = True  # False
adjust_threshold = True




# First step is to import the training and test sets
subsample_train = pd.read_csv('../data/mega_dfs/training.csv',sep='\t')
subsample_dev = pd.read_csv('../data/mega_dfs/dev.csv',sep='\t')
subsample_test = pd.read_csv('../data/mega_dfs/test.csv',sep='\t')

# Now load in the models
# It would be awesome to be able to do this in a list
# 'logistic/majority_hyper_and_energy.sav']
model_name_list = ['logistic/majority_hyper_and_energy.sav',
                   'logistic/baseline_hyper_and_energy.sav']
# 'kNN_hyper_and_energy_and_crs_2class_NN.sav']#kNN_all_2class



# print('in model directory', os.listdir('../models/logistic/'))
# print('knodle models', os.listdir('cache/'))


for name in model_name_list:
    # then its a knodle model and you need to load from cache:
    model = joblib.load('../models/'+name)
    # Load up the features
    # (not all saved models have features)
    try:
        feature_names = model.feature_names
    except:
        print('no features saved!')
        continue
    print('name', name)
    print(model.feature_names)

    try:
        print('model.model', model.model)

        # print('model._modules[linear]', model._modules['linear'])
    except:
        print('couldnt dig deeper')

    # First column is in this order: ['class hyper','class steve','class stowed']
    # hyper and steve are 1 if foreground
    # stowed is 1 if background, so it reverse activates
    
    # so model.data used to be the old input

    if model.normalizer == None:
        print('no norm')
        tensor_input = torch.from_numpy(subsample_test[model.feature_names].to_numpy())
    else:
       
        tensor_input = torch.from_numpy(
            model.normalizer.transform(subsample_test[model.feature_names]))
    
    
    try:
        #ys = model.model.predict(tensor_input)
        ys = model.model.forward(tensor_input)
    except AttributeError:
        try:
            ys = model.forward(tensor_input)
        except RuntimeError:
            ys = model.forward(tensor_input.float())
    print('ys', ys)
    print('detached', ys[:, 0].detach().numpy())
    STOP
            
    y_predicted = ys[:, 0].detach().numpy()
    
    compare_list = ['class overall','class hyper','class steve','class stowed']
    
    # Okay for a bunch of different ys, print stuff
    for comparison_thing in compare_list:
        print('comparing to this ~~~~~~ ', comparison_thing)
        
        y_actual = subsample_test[comparison_thing].values#).float()
        
        y_actual_tensor = np_array_to_tensor(y_actual).float()
        y_actual_tensor = y_actual_tensor.view(y_actual_tensor.shape[0],1)
        
        
        
        # accuracy with rounding
        acc = ys[:, 0].round().eq(y_actual_tensor.round()).sum() / float(y_actual.shape[0])
        print('test accuracy hyper', acc)
        
        TP, TN, FP, FN = calc_confusion(y_predicted.round(), y_actual.round())
        print('confusion')
        print('accuracy', (TP + TN)/(TP+TN+FP+FN))
        print('precision', TP / (TP + FP))
        print('recall', TP / (TP + FN))
    
    
    continue
    

    

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(y_predicted, bins=100, label='predicted', density=True)
    ax.hist(y_train, bins=4, label='actual', density=True)
    plt.legend()
    plt.show()
    
    

    try:
        print(model.model.eval(tensor_input))
    except AttributeError:
        print('eval doesnt work')
        #print(model.eval(tensor_input))

     
    predictions, dev_loss = model._prediction_loop(tensor_input)#, loss_calculation)

    print(model.model(tensor_input))
    # what about model.predict()?

    '''
     if isinstance(self.model, skorch.NeuralNetClassifier):
            # when the pytorch model is wrapped as a sklearn model (e.g. cleanlab)
            predictions = self.model.predict(dataset_to_numpy_input(features_dataset))
        else:
            feature_label_dataset = input_labels_to_tensordataset(features_dataset, gold_labels)
            feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)
            predictions, dev_loss = self._prediction_loop(feature_label_dataloader, loss_calculation)
    '''
   
    


    # Make a 3x2 for each classification before and after the filtering
    # How to pull the exact parameters tho?
    print(model.model_input_x)
    df = model.data
    

    # model.input_rules_matches_z
    in_rules = model.input_rules_matches_z
    out_rules = model.output_rules_matches_z

    



    

    # Okay time to go one column at a time
    df['before_hyper'] = in_rules[:,0]
    df['after_hyper'] = out_rules[:,0]

    df['before_steve'] = in_rules[:,1]
    df['after_steve'] = out_rules[:,1]

    df['before_stowed'] = in_rules[:,2]
    df['after_stowed'] = out_rules[:,2]

    try: #selecting out by ID
        df['id'] = model.ids
        print(model.ids)

        df = df[df['id'] == '1505']#hrciD2010-01-01bkgrndN0002.fits
    except:
        print('cannot select by ID')

    


    print('~~~~~~~~~~~ Diagnostics ~~~~~~~~~~~~~~~')
    print('length of df', len(df))
    print('~~~ Stowed ~~~~')
    print('length of real before', len(df[df['before_stowed']==0]))
    print('length of real after', len(df[df['after_stowed']==0]))

    print('~~~ Hyper ~~~~')
    print('length of real before', len(df[df['before_hyper']==1]))
    print('length of real after', len(df[df['after_hyper']==1]))

    print('~~~ Steve ~~~~')
    print('length of real before', len(df[df['before_steve']==1]))
    print('length of real after', len(df[df['after_steve']==1]))

    print('~~~ Stowed ~~~~')
    print('length of fake before', len(df[df['before_stowed']==1]))
    print('length of fake after', len(df[df['after_stowed']==1]))

    print('~~~ Hyper ~~~~')
    print('length of fake before', len(df[df['before_hyper']==0]))
    print('length of fake after', len(df[df['after_hyper']==0]))

    print('~~~ Steve ~~~~')
    print('length of fake before', len(df[df['before_steve']==0]))
    print('length of fake after', len(df[df['after_steve']==0]))

    print('~~~~~~~~~~~~~~~~~~~~~~')
    print('stats on what changed')
    print('fraction of stowed that change', len(df[((df['before_stowed']==0) & (df['after_stowed']==1)) | ((df['before_stowed']==1) & (df['after_stowed']==0))])/len(df))
    print('fraction of hyper that change', len(df[((df['before_hyper']==0) & (df['after_hyper']==1)) | ((df['before_hyper']==1) & (df['after_hyper']==0))])/len(df))
    print('fraction of steve that change', len(df[((df['before_steve']==0) & (df['after_steve']==1)) | ((df['before_steve']==1) & (df['after_steve']==0))])/len(df))


    
    # Make some comparison hyperbola figs

    xs_list = ['fp_u','fp_v','pha']
    ys_list = ['fb_u','fb_v','sumamps']
    for x, y in zip(xs_list, ys_list):

        
        fig = plt.figure(figsize = (10,4))
        ax = fig.add_subplot(131)
        ax.scatter(df[df['before_steve']==0][x].values, df[df['before_steve']==0][y].values, label='background', s=0.3, color='#14110F')
        ax.scatter(df[df['before_steve']==1][x].values, df[df['before_steve']==1][y].values, label='foreground', s=0.3, color='#FE5F55')
        ax.set_title('hyperbola before knodle')
        plt.legend()

        ax1 = fig.add_subplot(132)
        ax1.scatter(df[df['after_steve']==0][x].values, df[df['after_steve']==0][y].values, label='background', s=0.3, color='#14110F')
        ax1.scatter(df[df['after_steve']==1][x].values, df[df['after_steve']==1][y].values, label='foreground', s=0.3, color='#FE5F55')
        ax1.set_title('after knodle (total # = '+str(len(df))+')')
        plt.legend()

        ax2 = fig.add_subplot(133)
        ax2.scatter(df[(df['after_steve']==0) & (df['before_steve']==1)][x].values, df[(df['after_steve']==0) & (df['before_steve']==1)][y].values, 
            label='changed to background (# = '+str(len(df[(df['after_steve']==0) & (df['before_steve']==1)][x].values))+')', s=0.3, color='#14110F')
        ax2.scatter(df[(df['after_steve']==1) & (df['before_steve']==0)][x].values, df[(df['after_steve']==1) & (df['before_steve']==0)][y].values, 
            label='changed to foreground (# = '+str(len(df[(df['after_steve']==1) & (df['before_steve']==0)][x].values))+')', s=0.3, color='#FE5F55')
        ax2.set_title('what changed?')
        plt.legend()


        plt.show()

        

        
        fig = plt.figure(figsize = (10,4))
        ax2 = fig.add_subplot(131)
        ax2.scatter(df[df['before_hyper']==0][x].values, df[df['before_hyper']==0][y].values, label='background', s=0.3, color='#14110F')
        ax2.scatter(df[df['before_hyper']==1][x].values, df[df['before_hyper']==1][y].values, label='foreground', s=0.3, color='#FE5F55')
        ax2.set_title('hyperscreen before knodle')
        plt.legend()

        ax3 = fig.add_subplot(132)
        ax3.scatter(df[df['after_hyper']==0][x].values, df[df['after_hyper']==0][y].values, label='background', s=0.3, color='#14110F')
        ax3.scatter(df[df['after_hyper']==1][x].values, df[df['after_hyper']==1][y].values, label='foreground', s=0.3, color='#FE5F55')
        ax3.set_title('after knodle (total # = '+str(len(df))+')')
        plt.legend()

        ax4 = fig.add_subplot(133)
        ax4.scatter(df[(df['after_hyper']==0) & (df['before_hyper']==1)][x].values, df[(df['after_hyper']==0) & (df['before_hyper']==1)][y].values, 
            label='changed to background (# = '+str(len(df[(df['after_hyper']==0) & (df['before_hyper']==1)][y].values))+')', s=0.3, color='#14110F')
        ax4.scatter(df[(df['after_hyper']==1) & (df['before_hyper']==0)][x].values, df[(df['after_hyper']==1) & (df['before_hyper']==0)][y].values, 
            label='changed to foreground (# = '+str(len(df[(df['after_hyper']==1) & (df['before_hyper']==0)][y].values))+')', s=0.3, color='#FE5F55')
        ax4.set_title('what changed?')
        plt.legend()

        plt.show()

        
        fig = plt.figure(figsize = (10,4))
        ax2 = fig.add_subplot(131)
        ax2.scatter(df[df['before_stowed']==1][x].values, df[df['before_stowed']==1][y].values, label='background', s=0.3, color='#14110F')
        ax2.scatter(df[df['before_stowed']==0][x].values, df[df['before_stowed']==0][y].values, label='foreground', s=0.3, color='#FE5F55')
        ax2.set_title('stowed before knodle')
        plt.legend()

        ax3 = fig.add_subplot(132)
        ax3.scatter(df[df['after_stowed']==1][x].values, df[df['after_stowed']==1][y].values, label='background', s=0.3, color='#14110F')
        ax3.scatter(df[df['after_stowed']==0][x].values, df[df['after_stowed']==0][y].values, label='foreground', s=0.3, color='#FE5F55')
        ax3.set_title('after knodle (total # = '+str(len(df))+')')
        plt.legend()

        ax4 = fig.add_subplot(133)
        ax4.scatter(df[(df['after_stowed']==1) & (df['before_stowed']==0)][x].values, df[(df['after_stowed']==1) & (df['before_stowed']==0)][y].values, 
            label='changed to background (# = '+str(len(df[(df['after_stowed']==1) & (df['before_stowed']==0)][y].values))+')', s=0.3, color='#14110F')
        ax4.scatter(df[(df['after_stowed']==0) & (df['before_stowed']==1)][x].values, df[(df['after_stowed']==0) & (df['before_stowed']==1)][y].values, 
            label='changed to foreground (# = '+str(len(df[(df['after_stowed']==0) & (df['before_stowed']==1)][y].values))+')', s=0.3, color='#FE5F55')
        ax4.set_title('what changed?')
        plt.legend()

        plt.show()
    

    # Good that its shorter after
    nbins = 50
    # Make an image
    img_stowed_bg_before, yedges, xedges = np.histogram2d(df[df['before_stowed']==1]['y'].values, df[df['before_stowed']==1]['x'].values, nbins)#, range=extent)
    img_stowed_bg_after, yedges, xedges = np.histogram2d(df[df['after_stowed']==1]['y'].values, df[df['after_stowed']==1]['x'].values, nbins)#, range=extent)

    img_stowed_fg_before, yedges, xedges = np.histogram2d(df[df['before_stowed']==0]['y'].values, df[df['before_stowed']==0]['x'].values, nbins)#, range=extent)
    img_stowed_fg_after, yedges, xedges = np.histogram2d(df[df['after_stowed']==0]['y'].values, df[df['after_stowed']==0]['x'].values, nbins)#, range=extent)


    plt.clf()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(221)
    try:
        im = ax.imshow(abs(img_stowed_fg_before), norm=matplotlib.colors.LogNorm())
        ax.set_title('Foreground stowed before')
        plt.colorbar(im)
    except:
        im = ax.imshow(abs(img_stowed_fg_before))
        ax.set_title('Foreground stowed before')
        plt.colorbar(im)
    

    ax1 = fig.add_subplot(222)
    im1 = ax1.imshow(abs(img_stowed_fg_after), norm=matplotlib.colors.LogNorm())
    ax1.set_title('Foreground stowed after')
    plt.colorbar(im1)

    ax2 = fig.add_subplot(223)
    try:
        im2 = ax2.imshow(abs(img_stowed_bg_before), norm=matplotlib.colors.LogNorm())
        ax2.set_title('Background stowed before')
        plt.colorbar(im2)
    except:
        im2 = ax2.imshow(abs(img_stowed_bg_before))
        ax2.set_title('Background stowed before')
        plt.colorbar(im2)
    

    ax3 = fig.add_subplot(224)
    im3 = ax3.imshow(abs(img_stowed_bg_after), norm=matplotlib.colors.LogNorm())
    ax3.set_title('Background stowed after')
    plt.colorbar(im3)

    plt.show()

    img_hyper_bg_before, yedges, xedges = np.histogram2d(df[df['before_hyper']==1]['y'].values, df[df['before_hyper']==1]['x'].values, nbins)#, range=extent)
    img_hyper_bg_after, yedges, xedges = np.histogram2d(df[df['after_hyper']==1]['y'].values, df[df['after_hyper']==1]['x'].values, nbins)#, range=extent)

    img_hyper_fg_before, yedges, xedges = np.histogram2d(df[df['before_hyper']==0]['y'].values, df[df['before_hyper']==0]['x'].values, nbins)#, range=extent)
    img_hyper_fg_after, yedges, xedges = np.histogram2d(df[df['after_hyper']==0]['y'].values, df[df['after_hyper']==0]['x'].values, nbins)#, range=extent)


    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(221)
    im = ax.imshow(abs(img_hyper_fg_before), norm=matplotlib.colors.LogNorm())
    ax.set_title('Foreground hyper before')
    plt.colorbar(im)

    ax1 = fig.add_subplot(222)
    im1 = ax1.imshow(abs(img_hyper_fg_after), norm=matplotlib.colors.LogNorm())
    ax1.set_title('Foreground hyper after')
    plt.colorbar(im1)

    ax2 = fig.add_subplot(223)
    im2 = ax2.imshow(abs(img_hyper_bg_before), norm=matplotlib.colors.LogNorm())
    ax2.set_title('Background hyper before')
    plt.colorbar(im2)

    ax3 = fig.add_subplot(224)
    im3 = ax3.imshow(abs(img_hyper_bg_after), norm=matplotlib.colors.LogNorm())
    ax3.set_title('Background hyper after')
    plt.colorbar(im3)

    plt.show()

    img_steve_bg_before, yedges, xedges = np.histogram2d(df[df['before_steve']==1]['y'].values, df[df['before_steve']==1]['x'].values, nbins)#, range=extent)
    img_steve_bg_after, yedges, xedges = np.histogram2d(df[df['after_steve']==1]['y'].values, df[df['after_steve']==1]['x'].values, nbins)#, range=extent)

    img_steve_fg_before, yedges, xedges = np.histogram2d(df[df['before_steve']==0]['y'].values, df[df['before_steve']==0]['x'].values, nbins)#, range=extent)
    img_steve_fg_after, yedges, xedges = np.histogram2d(df[df['after_steve']==0]['y'].values, df[df['after_steve']==0]['x'].values, nbins)#, range=extent)


    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(221)
    im = ax.imshow(abs(img_steve_fg_before), norm=matplotlib.colors.LogNorm())
    ax.set_title('Foreground steve before')
    plt.colorbar(im)

    ax1 = fig.add_subplot(222)
    im1 = ax1.imshow(abs(img_steve_fg_after), norm=matplotlib.colors.LogNorm())
    ax1.set_title('Foreground steve after')
    plt.colorbar(im1)

    ax2 = fig.add_subplot(223)
    im2 = ax2.imshow(abs(img_steve_bg_before), norm=matplotlib.colors.LogNorm())
    ax2.set_title('Background steve before')
    plt.colorbar(im2)

    ax3 = fig.add_subplot(224)
    im3 = ax3.imshow(abs(img_steve_bg_after), norm=matplotlib.colors.LogNorm())
    ax3.set_title('Background steve after')
    plt.colorbar(im3)

    plt.show()

    

    STOP
    
    df_classify['random'] = np.random.random(len(df_classify))
    # Apply this model to the above data

    print('feature names', feature_names)
    comparison = df_classify.dropna(subset = feature_names)
    print('comparison', comparison)

    predict_this = comparison[feature_names]#[feature_list_RF]
    # data_unclassified['random'] = np.random.random(len(data_unclassified))



    try:
        predicted = model.model.predict(predict_this.values)
    except AttributeError:
        # This is because some of the saved models are actually NNs, 
        # so the way to run the prediction is to apply the model to the dataset
        print(predict_this)
        # Might be necessary to call
        # model.__modules['linear'].forward
        # print('Weight Of The Network :\n',model.model.weight)

        # print('Bias Of The Network :\n',model.model.bias)
        predicted = model.model(torch.tensor(predict_this.values))
    '''
    print('predicted output', predicted)
    print('numpy', predicted.detach().numpy()[0:10], type(predicted.detach().numpy()[0:10]))
    print('numpy', predicted.detach().numpy()[0:10,0])
    print('numpy list', list(predicted.detach().numpy())[0:10])
    '''
    comparison['predicted'] = predicted.detach().numpy()[:,0]
    
    plt.clf()
    height, bins= np.histogram(comparison['predicted'].values, bins=50, range=(-1,1))
    plt.hist(comparison['predicted'].values, bins)
    plt.xlabel('p values')
    plt.title(name)
    plt.show()
    

    # Make side by side histogram plots of the same data classified using hyperscreen and classified
    # using my new regression

    try:
        

        # Okay if that works what about adjusting the threshold until the percent accepted is most similar to hyper
        if adjust_threshold:
            thresh_list = np.linspace(-1,1,100)
            p_rejected = []
            for thresh in thresh_list:
                p_rejected.append(100 * len(comparison[comparison['predicted'] > thresh]) / len(comparison))
            print('percent rejected this method', p_rejected)
            print('percent rejected hyper', p_rejected_hyper)
            new_thresh = thresh_list[np.where(thresh_list == thresh_list.flat[np.abs(np.array(p_rejected) - 100*p_rejected_hyper).argmin()])[0][0]]
            data_pass = comparison[comparison['predicted'] < new_thresh]
            data_fail = comparison[comparison['predicted'] > new_thresh]
            print('threshold list', thresh_list)
            print('new thresh', new_thresh)
            print(round(100*len(data_fail)/len(comparison),2))
            print('supposed to match', p_rejected_hyper)
            new_thresh_list.append(thresh)
        else:
            data_pass = comparison[comparison['predicted'] < 0]
            data_fail = comparison[comparison['predicted'] > 0]

            
            

    except TypeError:
        # This means its an RFC not an RFR model, so the predictions are classes
        data_pass = comparison[comparison['predicted'] == 'pass_grant']
        data_fail = comparison[comparison['predicted'] == 'fail_grant']

        # Okay if that works what about adjusting the threshold until the percent accepted is most similar to hyper
        if adjust_threshold:
            # need to pull from predicted_proba
            
            predict_prob = model.predict_proba(predict_this.values)
            comparison['predicted_proba'] = predict_prob[:,1]

            thresh_list = np.linspace(0,1,100)
            p_rejected = []
            for thresh in thresh_list:
                p_rejected.append(100 * len(comparison[comparison['predicted_proba'] > thresh]) / len(comparison))
            
            new_thresh = thresh_list[np.where(thresh_list == thresh_list.flat[np.abs(p_rejected - p_rejected_hyper).argmin()])[0][0]]
            data_pass = comparison[comparison['predicted_proba'] < new_thresh]
            data_fail = comparison[comparison['predicted_proba'] > new_thresh]
            print('threshold list', thresh_list)
            print('new thresh', new_thresh)
            print(round(100*len(data_fail)/len(comparison),2))
            print('supposed to match', p_rejected_hyper)


    comparison_this_model.append(comparison)

    # print('comparison cols', comparison.columns)

    hyper_pass = comparison[comparison['class hyper'] < 0]
    hyper_fail = comparison[comparison['class hyper'] > 0]

    p_rejected_model_list.append(round(100*len(data_fail)/len(comparison),2))

    

    # % rejected as a function of radius (from center)
    center_x = np.mean(comparison['x'].values)
    center_y = np.mean(comparison['y'].values)
    extent = [[np.min(comparison['x'].values), np.max(comparison['x'].values)],
                  [np.min(comparison['y'].values), np.max(comparison['y'].values)]]

    nbins = 100
    img_data_all, yedges, xedges = np.histogram2d(comparison['y'].values, comparison['x'].values, nbins, range=extent)
    img_data_model_pass, yedges, xedges = np.histogram2d(data_pass['y'].values, data_pass['x'].values, nbins, range=extent)
    img_data_model_fail, yedges, xedges = np.histogram2d(data_fail['y'].values, data_fail['x'].values, nbins, range=extent)
    img_data_hyper_pass, yedges, xedges = np.histogram2d(hyper_pass['y'].values, hyper_pass['x'].values, nbins, range=extent)
    img_data_hyper_fail, yedges, xedges = np.histogram2d(hyper_fail['y'].values, hyper_fail['x'].values, nbins, range=extent)


    img_data_flat = np.minimum(img_data_all, 1)
    middle_pixel_x = np.shape(img_data_flat)[0]/2
    middle_pixel_y = np.shape(img_data_flat)[1]/2


    extent = np.shape(img_data_flat)[0]

    radial_bins_pix = np.linspace(0,extent/2 + extent/8,50)
    radial_bins_pix_flat = []
    radial_bins_pix_model_pass = []
    radial_bins_pix_model_fail = []
    radial_bins_pix_hyper_pass = []
    radial_bins_pix_hyper_fail = []

    for k in range(len(radial_bins_pix)-1):
        counter_flat = 0
        counter_model_pass = 0
        counter_model_fail = 0
        counter_hyper_pass = 0
        counter_hyper_fail = 0
        for i in range(np.shape(img_data_flat)[0]):
            for j in range(np.shape(img_data_flat)[1]):
                # so for each pixel see how far it is in pixel bins from the center
                if ((i - middle_pixel_x)**2 + (j - middle_pixel_y)**2 > radial_bins_pix[k]**2) & ((i - middle_pixel_x)**2 + (j - middle_pixel_y)**2 < radial_bins_pix[k+1]**2):
                    counter_flat+=img_data_flat[i,j] # only need to add 1 because its flat distribution
                    counter_model_pass+=img_data_model_pass[i,j]
                    counter_model_fail+=img_data_model_fail[i,j]
                    counter_hyper_pass+=img_data_hyper_pass[i,j]
                    counter_hyper_fail+=img_data_hyper_fail[i,j]
        radial_bins_pix_flat.append(counter_flat)
        radial_bins_pix_model_pass.append(counter_model_pass)
        radial_bins_pix_model_fail.append(counter_model_fail)
        radial_bins_pix_hyper_pass.append(counter_hyper_pass)
        radial_bins_pix_hyper_fail.append(counter_hyper_fail)
                    
    dist_flat = radial_bins_pix_flat/np.max(radial_bins_pix_flat)
    dist_model_pass = radial_bins_pix_model_pass/np.max(radial_bins_pix_model_pass)
    dist_model_fail = radial_bins_pix_model_fail/np.max(radial_bins_pix_model_fail)

    radial_rejected.append(radial_bins_pix_model_fail)

    # dist_model_pass_minus_fail

    dist_hyper_pass = radial_bins_pix_hyper_pass/np.max(radial_bins_pix_hyper_pass)
    dist_hyper_fail = radial_bins_pix_hyper_fail/np.max(radial_bins_pix_hyper_fail)


    hx_flat, hy_flat = margins(img_data_flat)
    hx_model_pass, hy_model_pass = margins(img_data_model_pass)
    hx_model_fail, hy_model_fail = margins(img_data_model_fail)
    hx_hyper_pass, hy_hyper_pass = margins(img_data_hyper_pass)
    hx_hyper_fail, hy_hyper_fail = margins(img_data_hyper_fail)

    # print('hx all', hx_all.T[0], 'hy all', hy_all[0])
    # hx_flat, hy_flat = img_data_flat.sum(axis=0), img_data_flat.sum(axis=1)
    # hx_all, hy_all = img_data_all.sum(axis=0), img_data_all.sum(axis=1)

    xs = np.linspace(0, len(hx_flat.T[0])-1, len(hx_flat.T[0]))
    ys = np.linspace(0, len(hy_flat[0])-1, len(hy_flat[0]))


    plt.clf()

    sns.set_style('darkgrid')

    fig = plt.figure(figsize=(15,5))
    ax0 = fig.add_subplot(111)
    ax0.step(radial_bins_pix[:-1], dist_flat, label='flat', color='black')
    ax0.step(radial_bins_pix[:-1], dist_model_pass, label='model pass', color='#210124')
    ax0.step(radial_bins_pix[:-1], dist_model_fail, label='model fail', color='#F8333C')
    ax0.step(radial_bins_pix[:-1], dist_hyper_pass, label='hyper pass', color='#750D37')
    ax0.step(radial_bins_pix[:-1], dist_hyper_fail, label='hyper fail', color='#BDBF09')

    plt.legend()
    ax0.set_xlabel('Distribution in r')
    ax0.set_title('Model is '+str(name)+', features = '+str(feature_names))
    plt.show()


    plt.clf()

    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(311)
    ax1.step(radial_bins_pix[:-1], dist_model_pass - dist_model_fail, label='model pass - fail', color='#F8333C')
    ax1.step(radial_bins_pix[:-1], dist_hyper_pass - dist_hyper_fail, label='hyper pass - fail', color='#BDBF09')
    ax1.set_xlabel('Normalized difference')
    ax1.legend()
    
    ax2 = fig.add_subplot(312)

    ax2.step(radial_bins_pix[:-1], radial_bins_pix_model_pass, label='model pass', color='#F8333C')
    ax2.step(radial_bins_pix[:-1], radial_bins_pix_hyper_pass, label='hyper pass', color='#BDBF09')
    ax2.set_xlabel('absolute number')
    ax2.legend()

    ax3 = fig.add_subplot(313)

    ax3.step(radial_bins_pix[:-1], radial_bins_pix_model_fail, label='model fail', color='#F8333C')
    ax3.step(radial_bins_pix[:-1], radial_bins_pix_hyper_fail, label='hyper fail', color='#BDBF09')
    ax3.set_xlabel('absolute number')
    ax3.legend()
    '''
    ax0.step(radial_bins_pix[:-1], dist_model_pass, label='model pass', color='#3A405A')
    ax0.step(radial_bins_pix[:-1], dist_model_fail, label='model fail', color='#AEC5EB')
    ax0.step(radial_bins_pix[:-1], dist_hyper_pass, label='hyper pass', color='#685044')
    ax0.step(radial_bins_pix[:-1], dist_hyper_fail, label='hyper fail', color='#E9AFA3')

    '''
    plt.tight_layout()
    plt.show()

plt.clf()
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
ax.step(radial_bins_pix[:-1], radial_bins_pix_hyper_fail, label='hyperscreen', color='black')

for i in range(len(model_name_list)):
    ax.step(radial_bins_pix[:-1], radial_rejected[i], label=model_name_list[i])
plt.legend()
plt.xlabel('radial bins')
plt.ylabel('number of events')
plt.title('failed events')
plt.show()

plt.clf()
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)

model_colors =['#0B3954','#087E8B','#BFD7EA','#FF5A5F','#C81D25']
print(radial_bins_pix, radial_bins_pix[:-1])
# = thresh_list[np.where(radial_bins_pix == radial_bins_pix.flat[np.abs(radial_bins_pix[:-1] - 10).argmin()])[0][0]]
starting_index = np.where(radial_bins_pix == radial_bins_pix.flat[np.abs(radial_bins_pix[:-1] - 10).argmin()])[0][0]
print('this index is closest to 10', starting_index)


for i in range(len(model_name_list)):
    diff = []
    for j in range(len(radial_rejected[i])):
        if j > starting_index:
            diff.append(radial_bins_pix_hyper_fail[j] - radial_rejected[i][j])
        else:
            diff.append(0)


    ax.step(radial_bins_pix[:-1], diff, label=model_name_list[i], color=model_colors[i])
    print(model_name_list[i], np.sum(diff))
    print('normalized by number of bins', np.sum(diff)/len(radial_bins_pix_hyper_fail))
plt.legend()
plt.xlabel('radial bins')
plt.ylabel('hyper - num of events')
plt.title('Fail hyper - fail model')
plt.show()






print('percent rejected hyperscreen', p_rejected_hyper)
print('models', model_name_list)
print('percent rejected model', p_rejected_model_list, 'new thresh', )

# Can I now make side by side plots of hyper and all the other classifications?
sns.set_style('white')
plt.clf()


nbins=100

hyper_pass = comparison_this_model[0][comparison_this_model[0]['class hyper'] < 0]
hyper_fail = comparison_this_model[0][comparison_this_model[0]['class hyper'] > 0]



img_hyper_pass, yedges, xedges = np.histogram2d(hyper_pass['y'].values, hyper_pass['x'].values, nbins)#, range=extent)
img_hyper_fail, yedges, xedges = np.histogram2d(hyper_fail['y'].values, hyper_fail['x'].values, nbins)#, range=extent)


if adjust_threshold:
    try:
        data_pass_0 = comparison_this_model[0][comparison_this_model[0]['predicted_proba'] < new_thresh_list[0]]
        data_fail_0 = comparison_this_model[0][comparison_this_model[0]['predicted_proba'] > new_thresh_list[0]]
    except:
        data_pass_0 = comparison_this_model[0][comparison_this_model[0]['predicted'] < new_thresh_list[0]]
        data_fail_0 = comparison_this_model[0][comparison_this_model[0]['predicted'] > new_thresh_list[0]]
else:
    try:
        data_pass_0 = comparison_this_model[0][comparison_this_model[0]['predicted'] < 0]
        data_fail_0 = comparison_this_model[0][comparison_this_model[0]['predicted'] > 0]
    except:
        data_pass_0 = comparison_this_model[0][comparison_this_model[0]['predicted'] == 'pass_grant']
        data_fail_0 = comparison_this_model[0][comparison_this_model[0]['predicted'] == 'fail_grant']
        

img_model_0_pass, yedges, xedges = np.histogram2d(data_pass_0['y'].values, data_pass_0['x'].values, nbins)#, range=extent)
img_model_0_fail, yedges, xedges = np.histogram2d(data_fail_0['y'].values, data_fail_0['x'].values, nbins)#, range=extent)

print('length of pass', len(data_pass_0), 'length of fail', len(data_fail_0))

if adjust_threshold:
    try:
        data_pass_1 = comparison_this_model[1][comparison_this_model[1]['predicted_proba'] < new_thresh_list[1]]
        data_fail_1 = comparison_this_model[1][comparison_this_model[1]['predicted_proba'] > new_thresh_list[1]]
    except:
        try:
            data_pass_1 = comparison_this_model[1][comparison_this_model[1]['predicted'] < new_thresh_list[1]]
            data_fail_1 = comparison_this_model[1][comparison_this_model[1]['predicted'] > new_thresh_list[1]]
        except IndexError: # Which means htere is no model 1
            data_pass_1 = data_pass_0
            data_fail_1 = data_fail_0
else:
    try:
        data_pass_1 = comparison_this_model[1][comparison_this_model[1]['predicted'] < 0]
        data_fail_1 = comparison_this_model[1][comparison_this_model[1]['predicted'] > 0]
    except:
        try:
            data_pass_1 = comparison_this_model[1][comparison_this_model[1]['predicted'] == 'pass_grant']
            data_fail_1 = comparison_this_model[1][comparison_this_model[1]['predicted'] == 'fail_grant']
        except IndexError: # Which means htere is no model 1
            data_pass_1 = data_pass_0
            data_fail_1 = data_fail_0
img_model_1_pass, yedges, xedges = np.histogram2d(data_pass_1['y'].values, data_pass_1['x'].values, nbins)#, range=extent)
img_model_1_fail, yedges, xedges = np.histogram2d(data_fail_1['y'].values, data_fail_1['x'].values, nbins)#, range=extent)




fig = plt.figure(figsize=(10,8))
ax0 = fig.add_subplot(231)
ax0.imshow(img_hyper_pass, norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**4))
ax0.set_title('Hyper pass')

ax2 = fig.add_subplot(232)
ax2.imshow(img_model_0_pass, norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**4))
ax2.set_title('Model '+str(model_name_list[0])+' pass')

ax4 = fig.add_subplot(233)
ax4.imshow(img_model_1_pass, norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**4))
try:
    ax4.set_title('Model '+str(model_name_list[1])+' pass')
except IndexError:
    ax4.set_title('No Model 1')

ax1 = fig.add_subplot(234)
ax1.imshow(img_hyper_fail, norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**4))
ax1.set_title('Fail ('+str(p_rejected_hyper)+'p rejected)')


ax3 = fig.add_subplot(235)
ax3.imshow(img_model_0_fail, norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**4))
ax3.set_title('Fail ('+str(p_rejected_model_list[0])+'p rejected)')



ax5 = fig.add_subplot(236)
ax5.imshow(img_model_1_fail, norm=matplotlib.colors.LogNorm(vmin=1, vmax=10**4))
try:
    ax5.set_title('Fail ('+str(p_rejected_model_list[1])+'p rejected)')
except:
    ax5.set_title('No Model 1')
plt.show()

STOP

# Now make scatterplots of pha, sumamps, and hte hyperbola axes with also six panels of pass fail :)

x_list = ['fp_u','fp_v','pha']
y_list = ['fb_u','fb_v','sumamps']

sns.set_style('darkgrid')

for i in range(len(x_list)):
    x = x_list[i]
    y = y_list[i]

    # Find max and min
    min_plot_x = np.min([np.min(hyper_pass[x].values),np.min(hyper_fail[x].values)])
    max_plot_x = np.max([np.max(hyper_pass[x].values),np.max(hyper_fail[x].values)])

    min_plot_y = np.min([np.min(hyper_pass[y].values),np.min(hyper_fail[y].values)])
    max_plot_y = np.max([np.max(hyper_pass[y].values),np.max(hyper_fail[y].values)])

    plt.clf()
    fig = plt.figure(figsize=(10,8))
    ax0 = fig.add_subplot(231)
    ax0.scatter(hyper_pass[x].values, hyper_pass[y].values, color = '#F8333C', s=0.2)
    ax0.set_title('Hyper pass')
    ax0.set_xlim([min_plot_x, max_plot_x])
    ax0.set_ylim([min_plot_y, max_plot_y])

    ax1 = fig.add_subplot(234)
    ax1.scatter(hyper_fail[x].values, hyper_fail[y].values, color = '#BDBF09', s=0.2)
    ax1.set_title('Fail ('+str(p_rejected_hyper)+'p rejected)')
    ax1.set_xlim([min_plot_x, max_plot_x])
    ax1.set_ylim([min_plot_y, max_plot_y])

    ax2 = fig.add_subplot(232)
    ax2.scatter(data_pass_0[x].values, data_pass_0[y].values, color = '#F8333C', s=0.2)
    ax2.set_title('Model '+str(model_name_list[0])+' pass')
    ax2.set_xlim([min_plot_x, max_plot_x])
    ax2.set_ylim([min_plot_y, max_plot_y])

    ax3 = fig.add_subplot(235)
    ax3.scatter(data_fail_0[x].values, data_fail_0[y].values, color = '#BDBF09', s=0.2)
    ax3.set_title('Fail ('+str(p_rejected_model_list[0])+'p rejected)')
    ax3.set_xlim([min_plot_x, max_plot_x])
    ax3.set_ylim([min_plot_y, max_plot_y])

    ax4 = fig.add_subplot(233)
    ax4.scatter(data_pass_1[x].values, data_pass_1[y].values, color = '#F8333C', s=0.2)
    ax4.set_title('Model '+str(model_name_list[1])+' pass')
    ax4.set_xlim([min_plot_x, max_plot_x])
    ax4.set_ylim([min_plot_y, max_plot_y])

    ax5 = fig.add_subplot(236)
    ax5.scatter(data_fail_1[x].values, data_fail_1[y].values, color = '#BDBF09', s=0.2)
    ax5.set_title('Fail ('+str(p_rejected_model_list[1])+'p rejected)')
    ax5.set_xlim([min_plot_x, max_plot_x])
    ax5.set_ylim([min_plot_y, max_plot_y])

    plt.show()

'''
ax0 = fig.add_subplot(131)
ax0.step(xs, hx_flat.T[0]/np.max(hx_flat.T[0]), label='Flat', color='#59CD90')
ax0.step(xs, hx_model_.T[0]/np.max(hx_all.T[0]),  label='Cas A', color='#EE6352')#, density=True)
ax0.step(xs, hx_select.T[0]/np.max(hx_select.T[0]), label='Select', color='#3FA7D6')
plt.legend()
ax0.set_title('Distribution in x')

ax1 = fig.add_subplot(132)
ax1.step(ys, hy_flat[0]/np.max(hy_flat[0]), label='Flat', color='#59CD90')
ax1.step(ys, hy_all[0]/np.max(hy_all[0]),  label='Cas A', color='#EE6352')#, density=True)
ax1.step(ys, hy_select[0]/np.max(hy_select[0]), label='Select', color='#3FA7D6')
plt.legend()
ax1.set_title('Distribution in y')


ax2 = fig.add_subplot(133)
ax2.step(radial_bins_pix[:-1], dist_flat, label='Flat', color='#59CD90')
ax2.step(radial_bins_pix[:-1], dist_all,  label='Cas A', color='#EE6352')#, density=True)
ax2.step(radial_bins_pix[:-1], dist_sample, label='Select', color='#3FA7D6')
plt.legend()
ax2.set_title('Distribution in r')
'''





STOP

if plot:

    x_list = ['x','fp_u','fp_v','pha']
    y_list = ['y','fb_u','fb_v','sumamps']

    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        plt.clf()
        plt.scatter(subsample[x].values,
                    subsample[y].values,
                   c=subsample['class'],  s=0.5)
        plt.colorbar()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

# Run this through a RFR 
# (because now you have numbers instead of a binary classification)

# Is there a way to subsample more evenly? i.e., same number from each class?
feature_list_RF = ['fp_u','fb_u','fp_v','fb_v','pha','sumamps','random']
number = 1000
bg_chopped = df[df['class'] == 1].sample(n=number)
hyper_bg_chopped = df[df['class'] == 0.5].sample(n=number)
hyper_fg_chopped = df[df['class'] == -0.5].sample(n=number)

# Put this all together into one df
classify = pd.concat([bg_chopped, hyper_bg_chopped, hyper_fg_chopped])
classify['random'] = np.random.random(len(classify))

classify = classify[['fp_u','fb_u','fp_v','fb_v','pha','sumamps','random','class']].dropna()

if plot:

    x_list = ['x','fp_u','fp_v','pha']
    y_list = ['y','fb_u','fb_v','sumamps']

    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        plt.clf()
        plt.scatter(classify[x].values,
                    classify[y].values,
                   c=classify['class'],  s=0.5)
        plt.colorbar()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

suffix = 'avs'
most_imp, unimportant, model  = utils_ML.run_RFR(classify, 
    feature_list_RF, '../models/RFR_'+str(suffix), 
    verbose=True, hyper_fit = False, refit = False)

# Okay now try to see how it worked
# compare 1505 between both
data = df[df['id']==1505]
data = data.dropna(subset = ['x','y','fp_u','fp_v','fb_u','fb_v','pha','sumamps'])
print(data)

if plot:

    x_list = ['x','fp_u','fp_v','pha']
    y_list = ['y','fb_u','fb_v','sumamps']

    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        plt.clf()
        plt.scatter(data[x].values,
                    data[y].values,
                   c=data['class'],  s=0.5)
        plt.colorbar()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

# Now do some classification
# data_unclassified = data.reset_index()
data['random'] = np.random.random(len(data))
predict_this = data[feature_list_RF]
# data_unclassified['random'] = np.random.random(len(data_unclassified))
predicted = model.predict(predict_this.values)
# predict_prob = model.predict_proba(data_unclassified[feature_list_RF].dropna().values)

print(predicted)

data['predicted'] = predicted

print(data['predicted'])

# Make side by side histogram plots of the same data classified using hyperscreen and classified
# using my new regression

data_pass = data[data['predicted'] < 0]
data_fail = data[data['predicted'] > 0]

data_pass_hyper = data[data['class'] == -0.5]
data_fail_hyper = data[data['class'] == 0.5]


# data_pass = data_unclassified[data_unclassified['predicted'] < 0]
# data_fail = data_unclassified[data_unclassified['predicted'] > 0]

print('length lt 0 so real', len(data_pass))
print('length gt 0 so background', len(data_fail))

if plot:
    x_list = ['fp_u','fp_v','pha']
    y_list = ['fb_u','fb_v','sumamps']

    for i in range(len(x_list)):

        

        x = x_list[i]
        y = y_list[i]
        plt.clf()
        plt.scatter(data_fail[x].values,
                    data_fail[y].values,
                    s=0.5, label='fail')
        plt.scatter(data_pass[x].values,
                    data_pass[y].values,
                   s=0.5, label='pass')
        plt.legend()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()


print('make sure these lengths are the same', len(data), len(data_pass))
print('headers of data', data.columns)

nbins = 100


plt.clf()
fig = plt.figure()
ax0 = fig.add_subplot(221)

img_data, yedges, xedges = np.histogram2d(data_pass['y'].values, data_pass['x'].values, nbins)
im0 = ax0.imshow(img_data, norm=matplotlib.colors.LogNorm())
ax0.set_title('RFR foreground (# = '+str(len(data_pass['y'].values))+')')
ax0.axis('off')
plt.colorbar(im0)

ax1 = fig.add_subplot(222)

img_data, yedges, xedges = np.histogram2d(data_fail['y'].values, data_fail['x'].values, nbins)
im1 = ax1.imshow(img_data, norm=matplotlib.colors.LogNorm())
ax1.set_title('RFR background (# = '+str(len(data_fail['y'].values))+')')
ax1.axis('off')
plt.colorbar(im1)

ax2 = fig.add_subplot(223)

img_data, yedges, xedges = np.histogram2d(data_pass_hyper['y'].values, data_pass_hyper['x'].values, nbins)
im2 = ax2.imshow(img_data, norm=matplotlib.colors.LogNorm())
ax2.set_title('Hyper foreground (# = '+str(len(data_pass_hyper['y'].values))+')')
ax2.axis('off')
plt.colorbar(im2)

ax3 = fig.add_subplot(224)

img_data, yedges, xedges = np.histogram2d(data_fail_hyper['y'].values, data_fail_hyper['x'].values, nbins)
im3 = ax3.imshow(img_data, norm=matplotlib.colors.LogNorm())
ax3.set_title('Hyper background (# = '+str(len(data_fail_hyper['y'].values))+')')
ax3.axis('off')
plt.colorbar(im3)

plt.show()


# Now do a thresholding analysis
same = True
threshold_list = [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for thresh in threshold_list:
    data_pass = data[data['predicted'] < thresh]
    data_fail = data[data['predicted'] > thresh]

    print(thresh)
    print('length of pass', len(data_pass))
    print('length of fail', len(data_fail))

    if len(data_pass)==0 or len(data_fail)==0:
        continue

    plt.clf()
    fig = plt.figure()
    ax0 = fig.add_subplot(121)

    img_data_fore, yedges, xedges = np.histogram2d(data_pass['y'].values, data_pass['x'].values, nbins)
    img_data_bk, yedges, xedges = np.histogram2d(data_fail['y'].values, data_fail['x'].values, nbins)

    if same:
        im0 = ax0.imshow(img_data_fore, norm=matplotlib.colors.LogNorm(vmin = 1, vmax = 10**4))
        ax0.set_title('RFR foreground (# = '+str(len(data_pass['y'].values))+')')
        ax0.axis('off')
        ax0.annotate(str(thresh), xy=(0.02,0.07), xycoords='axes fraction')
        plt.colorbar(im0, fraction=0.045)

        ax1 = fig.add_subplot(122)

        im1 = ax1.imshow(img_data_bk, norm=matplotlib.colors.LogNorm(vmin = 1, vmax = 10**4))
        ax1.set_title('RFR background (# = '+str(len(data_fail['y'].values))+')')
        ax1.axis('off')
        plt.colorbar(im1, fraction=0.045)
        # plt.title()
        plt.show()
        continue

    min = np.min([np.min(img_data_fore),np.min(img_data_bk)])
    max = np.max([np.max(img_data_fore),np.max(img_data_bk)])

    print('min', min, 'max', max)


    if min==0.0:
        im0 = ax0.imshow(img_data_fore, norm=matplotlib.colors.LogNorm(vmin = 1, vmax = max))
        ax0.set_title('RFR foreground (# = '+str(len(data_pass['y'].values))+')')
        ax0.axis('off')
        ax0.annotate(str(thresh), xy=(0.02,0.07), xycoords='axes fraction')
        plt.colorbar(im0, fraction=0.045)

        ax1 = fig.add_subplot(122)

        im1 = ax1.imshow(img_data_bk, norm=matplotlib.colors.LogNorm(vmin = 1, vmax = max))
        ax1.set_title('RFR background (# = '+str(len(data_fail['y'].values))+')')
        ax1.axis('off')
        plt.colorbar(im1, fraction=0.045)
        # plt.title()
        plt.show()
    else:
    
        im0 = ax0.imshow(img_data_fore, norm=matplotlib.colors.LogNorm(vmin = min, vmax = max))
        ax0.set_title('RFR foreground (# = '+str(len(data_pass['y'].values))+')')
        ax0.axis('off')
        ax0.annotate(str(thresh), xy=(0.02,0.07), xycoords='axes fraction')
        plt.colorbar(im0, fraction=0.045)

        ax1 = fig.add_subplot(122)

        im1 = ax1.imshow(img_data_bk, norm=matplotlib.colors.LogNorm(vmin = min, vmax = max))
        ax1.set_title('RFR background (# = '+str(len(data_fail['y'].values))+')')
        ax1.axis('off')
        plt.colorbar(im1, fraction=0.045)
        # plt.title()
        plt.show()


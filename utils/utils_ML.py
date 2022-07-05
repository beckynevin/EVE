import hyperscreen
#from hyperscreen import hypercore, hypercore_csv, hypercore_stowed
import pandas as pd
#from pulearn import BaggingPuClassifier
import seaborn as sns
import numpy.ma as ma
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import joblib
from sklearn.model_selection import RandomizedSearchCV
import torch
from torch.utils.data import TensorDataset
import numpy as np
import scipy.sparse as sp


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

def testAndTrainIndices(test_fold, Nfolds, folds):

    print('finding test and train indices...')

    train_folds = np.delete(np.arange(Nfolds), test_fold)

    test_ind   = [i for i in range(len(folds)) if folds[i]==test_fold]
    train_ind  = [i for i in range(len(folds)) if folds[i] in train_folds]

    return test_ind, train_ind

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def run_QDA(df_merg, features_list,  model_name, verbose, hyper_fit=False, hyper = None):
    Nfolds = 10
    Ndat = 5000

    features = df_merg[features_list].values
    #,'nspax','re'
    Nfeatures = len(features[0])

    labels = df_merg[['class']].values
    folds = np.arange(len(labels))%Nfolds
    
    
    #Test on fold 0, train on the remaining folds:
    test_ind, train_ind = testAndTrainIndices(test_fold = 0, Nfolds = Nfolds, folds=folds)
    
    #divide features and labels into test and train sets:
    test_features = features[test_ind]
    test_labels   = labels[test_ind]
   
    train_features  = features[train_ind]
    train_labels    = labels[train_ind]

    model = QuadraticDiscriminantAnalysis(store_covariance=True)
    print('params base model')
    print(model.get_params())
    model.fit(train_features, train_labels)

    return model

def run_RFR(df_merg, features_list,  model_name, verbose, hyper_fit=False, hyper = False, save = True):
    # These are adjustable RFR parameters
    Nfolds = 10
    Ndat = 5000

    features = df_merg[features_list].values
    #,'nspax','re'
    Nfeatures = len(features[0])

    print('# of featres', Nfeatures)
    
    #dat['features']#.reshape(-1,1)
    labels = df_merg[['class']].values
    folds = np.arange(len(labels))%Nfolds
    
    
    #Test on fold 0, train on the remaining folds:
    test_ind, train_ind = testAndTrainIndices(test_fold = 0, Nfolds = Nfolds, folds=folds)
    
    #divide features and labels into test and train sets:
    test_features = features[test_ind]
    test_labels   = labels[test_ind]
   
    train_features  = features[train_ind]
    train_labels    = labels[train_ind]

    # first check to see if the model already exists:
    filename = model_name+'.sav'
    print('filename', filename)
    print('does this exist', os.path.exists(filename))

    if os.path.exists(filename) and hyper == False and save == False:
        print('loading existing model')
        model = joblib.load(filename)
    else:
        # Then fit and save the model
        #make a random forest model:
        

        if hyper_fit:
            model = RandomForestRegressor(max_depth=10, random_state=42)
            print('params base model')
            print(model.get_params())
            model.fit(train_features, train_labels)

            print('base accuracy')
            print(model.score(test_features, test_labels))


            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}


            model_random = RandomizedSearchCV(estimator = model, 
                param_distributions = random_grid, 
                n_iter = 100, # was 100
                cv = 3, verbose=2, random_state=42, n_jobs = -1)
            model_random.fit(train_features, train_labels)
            print('best params')
            print(model_random.best_params_)
            print('best params accuracy')
            print(model_random.score(test_features, test_labels))
            
            hyper = model_random.best_params_
            model = RandomForestRegressor(n_estimators = hyper['n_estimators'],
                    min_samples_split = hyper['min_samples_split'], 
                    min_samples_leaf = hyper['min_samples_leaf'], 
                    max_features = hyper['max_features'],
                    max_depth = hyper['max_depth'],
                    bootstrap = hyper['bootstrap'])
         
            model.fit(train_features, train_labels)
            print(model.get_params())
            print('best accuracy')
            print(model.score(test_features, test_labels))
            
        else:
            if hyper:
                model = RandomForestRegressor(n_estimators = hyper['n_estimators'],
                    min_samples_split = hyper['min_samples_split'], 
                    min_samples_leaf = hyper['min_samples_leaf'], 
                    max_features = hyper['max_features'],
                    max_depth = hyper['max_depth'],
                    bootstrap = hyper['bootstrap'])
                model.fit(train_features, train_labels.ravel())
                print(model.get_params())
                print('base accuracy')
                print(model.score(test_features, test_labels))
                
            else: # just fit the default model
                model = RandomForestRegressor(max_depth=10, random_state=42)

                model.fit(train_features, train_labels)
        # save the model to disk
        if save:
            model.feature_names = features_list
            joblib.dump(model, filename)

    print('predicting...')
    # Predict on new data
    preds = model.predict(test_features)
    #print out the first few mass predictions to see if they make sense:
    if verbose:
        for h in range(10):
            print(test_labels[h], preds[h])

    #print('made it through creating model', preds)
    # rank feature importance:
    importances = model.feature_importances_
    #print('ranked importances', importances)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    #print('std and indices', std, indices)

    if verbose:
        # Plot the feature importances of the forest
        plt.clf()
        plt.figure(figsize=(15,5))
        #plt.title("RFR Feature importances for "+str(run))
        plt.bar(range(Nfeatures), importances[indices], yerr=std[indices], align="center", color='pink')
        plt.xticks(range(Nfeatures), indices)
        plt.xlim([-1, Nfeatures])
        plt.show()
        
        #plt.savefig('feature_importance_'+str(run)+'_rando.pdf')
        
        
        
        print('Importance in Order ~~~~')
    
    # find the index of the random one:
    random_idx = features_list.index('random')
    random_value = importances[random_idx]
    random_std = std[random_idx]
    if verbose:
        print('random idx', random_idx)
        print('random_value', random_value)
    unin_here = []
    important_here = []
    for j in range(len(indices)):
        #if importances[indices[j]] - std[indices[j]] > 0:
        print(indices[j], features_list[indices[j]])
        if importances[indices[j]] > random_value:# or importances[indices[j]] - std[indices[j]] > random_value - random_std:
            important_here.append(features_list[indices[j]])
        else:
            unin_here.append(features_list[indices[j]])
        
  
    return important_here, unin_here, model



def run_RFC(df_merg, features_list,  model_name, verbose, hyper_fit=False, hyper = None, save = True):
    # These are adjustable RFR parameters
    Nfolds = 10
    Ndat = 5000

    features = df_merg[features_list].values
    #,'nspax','re'
    Nfeatures = len(features[0])

    print('# of featres', Nfeatures)
    
    #dat['features']#.reshape(-1,1)
    labels = df_merg[['class']].values
    folds = np.arange(len(labels))%Nfolds
    
    
    #Test on fold 0, train on the remaining folds:
    test_ind, train_ind = testAndTrainIndices(test_fold = 0, Nfolds = Nfolds, folds=folds)
    
    #divide features and labels into test and train sets:
    test_features = features[test_ind]
    test_labels   = labels[test_ind]
   
    train_features  = features[train_ind]
    train_labels    = labels[train_ind]

    # first check to see if the model already exists:
    filename = model_name+'.sav'

    if os.path.exists(filename) and hyper == False and save == False:
        print('loading existing model')
        model = joblib.load(filename)
    else:
        # Then fit and save the model
        #make a random forest model:
        

        if hyper_fit:
            model = RandomForestClassifier(max_depth=10, random_state=42)
            print('params base model')
            print(model.get_params())
            model.fit(train_features, train_labels)

            print('base accuracy')
            print(model.score(test_features, test_labels))


            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}


            model_random = RandomizedSearchCV(estimator = model, 
                param_distributions = random_grid, 
                n_iter = 100, 
                cv = 3, verbose=2, random_state=42, n_jobs = -1)
            model_random.fit(train_features, train_labels)
            print('best params')
            print(model_random.best_params_)
            print('best params accuracy')
            print(model_random.score(test_features, test_labels))
            
            hyper = model_random.best_params_
            model = RandomForestClassifier(n_estimators = hyper['n_estimators'],
                    min_samples_split = hyper['min_samples_split'], 
                    min_samples_leaf = hyper['min_samples_leaf'], 
                    max_features = hyper['max_features'],
                    max_depth = hyper['max_depth'],
                    bootstrap = hyper['bootstrap'])
            model.fit(train_features, train_labels)
            print(model.get_params())
            print('best accuracy')
            print(model.score(test_features, test_labels))
            
        else:
            if hyper:
                model = RandomForestClassifier(n_estimators = hyper['n_estimators'],
                    min_samples_split = hyper['min_samples_split'], 
                    min_samples_leaf = hyper['min_samples_leaf'], 
                    max_features = hyper['max_features'],
                    max_depth = hyper['max_depth'],
                    bootstrap = hyper['bootstrap'])
                model.fit(train_features, train_labels.ravel())
                print(model.get_params())
                print('base accuracy')
                print(model.score(test_features, test_labels))
                
            else: # just fit the default model
                model = RandomForestClassifier(max_depth=10, random_state=42)

                model.fit(train_features, train_labels)
        # save the model to disk
        if save:
            model.feature_names = features_list
            joblib.dump(model, filename)

    print('predicting...')
    # Predict on new data
    preds = model.predict(test_features)
    #print out the first few mass predictions to see if they make sense:
    if verbose:
        for h in range(10):
            print(test_labels[h], preds[h])

    #print('made it through creating model', preds)
    # rank feature importance:
    importances = model.feature_importances_
    #print('ranked importances', importances)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    #print('std and indices', std, indices)

    if verbose:
        # Plot the feature importances of the forest
        plt.clf()
        plt.figure(figsize=(15,5))
        #plt.title("RFR Feature importances for "+str(run))
        plt.bar(range(Nfeatures), importances[indices], yerr=std[indices], align="center", color='pink')
        plt.xticks(range(Nfeatures), indices)
        plt.xlim([-1, Nfeatures])
        plt.show()
        
        #plt.savefig('feature_importance_'+str(run)+'_rando.pdf')
        
        
        
        print('Importance in Order ~~~~')
    
    # find the index of the random one:
    random_idx = features_list.index('random')
    random_value = importances[random_idx]
    random_std = std[random_idx]
    if verbose:
        print('random idx', random_idx)
        print('random_value', random_value)
    unin_here = []
    important_here = []
    for j in range(len(indices)):
        #if importances[indices[j]] - std[indices[j]] > 0:
        print(indices[j], features_list[indices[j]])
        if importances[indices[j]] > random_value:# or importances[indices[j]] - std[indices[j]] > random_value - random_std:
            important_here.append(features_list[indices[j]])
        else:
            unin_here.append(features_list[indices[j]])
        
  
    return important_here, unin_here, model





def plot_background(input_data, n_points, name):
    
    #bkg_filt[1].data['x']
    x = input_data[1].data['x']#[0:n_points]#[np.array(binary_preds)==1]#model_lda.predict(xs)
    y = input_data[1].data['y']#[0:n_points]#[np.array(binary_preds)==1]

    nbins=100
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #print('# Real events: ', len(input_data.loc[good_hyperbola]), '# All events: ',len(input_data))
    #print('# bg events: ', len(input_data)-len(input_data.loc[good_hyperbola]))
    plt.clf()
    im = plt.imshow(img_data,  rasterized=True, cmap='viridis', origin='data', extent=extent)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(im)
    plt.title(name+' Background')
    plt.show()
    
def plot_science(input_data, name):
    
    #bkg_filt[1].data['x']
    x = input_data[1].data['x']#[0:n_points]#[np.array(binary_preds)==1]#model_lda.predict(xs)
    y = input_data[1].data['y']#[0:n_points]#[np.array(binary_preds)==1]

    nbins=100
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #print('# Real events: ', len(input_data.loc[good_hyperbola]), '# All events: ',len(input_data))
    #print('# bg events: ', len(input_data)-len(input_data.loc[good_hyperbola]))
    plt.clf()
    im = plt.imshow(ma.masked_where(img_data==0, img_data),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(im)
    plt.title(name+' Science')
    plt.show()
    
def plot_from_df(df, name, var1, var2):
    df = df.dropna()
    x = df[var1]#[0:n_points]#[np.array(binary_preds)==1]#model_lda.predict(xs)
    y = df[var2]#[0:n_points]#[np.array(binary_preds)==1]

    
    
    nbins=100#int(len(x)/10)
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #print('# Real events: ', len(input_data.loc[good_hyperbola]), '# All events: ',len(input_data))
    #print('# bg events: ', len(input_data)-len(input_data.loc[good_hyperbola]))
    try:
        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel(var1)
        plt.ylabel(var2)
    
        plt.colorbar()
        plt.title(name)
        #plt.axes().set_aspect('equal', 'datalim')
        if var1=='pha':
            plt.xlim([0,270])
            plt.axes().set_aspect(10)
        plt.show()
    except:
        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.colorbar()
        plt.title(name)
        #plt.axes().set_aspect('equal', 'datalim')
        if var1=='pha':
            plt.xlim([0,270])
            plt.axes().set_aspect(10)
        plt.show()
    
    
    '''plt.clf()
    plt.scatter(x, y, c = df['pha'], cmap='viridis')
    plt.colorbar()
    plt.title(name)
    plt.show()'''
    
    
def kde_plots(input_data, xaxis, yaxis):
    feature_list = [xaxis, yaxis]
    X = pd.DataFrame(input_data[feature_list].values, columns=feature_list)
    
    
    plt.clf()
    sns.kdeplot(data = X, 
            x=xaxis, y=yaxis, fill=True)
    plt.show()
    
def scatter_plots(input_data, xaxis, yaxis, label):
    feature_list = [xaxis, yaxis]
    try:
        X = pd.DataFrame(input_data[feature_list].values, columns=feature_list)
    except:
        X = input_data[feature_list].values
    
    plt.clf()
    sns.scatterplot(data = X, 
            x=xaxis, y=yaxis, s=1)
    plt.title(label)
    plt.show()
def plot_bagging_results(X, y, results_bag, xaxis, yaxis, idx_zero, idx_one, thresh):
    # plot the results
    plt.clf()
    fig = plt.figure(figsize=(12,5))
    ax0 = fig.add_subplot(121)
    
    xmin = np.min([np.min(X[xaxis].values[idx_zero]),np.min(X[xaxis].values[idx_one])])
    xmax = np.max([np.max(X[xaxis].values[idx_zero]),np.max(X[xaxis].values[idx_one])])
    
    ymin = np.min([np.min(X[yaxis].values[idx_zero]),np.min(X[yaxis].values[idx_one])])
    ymax = np.max([np.max(X[yaxis].values[idx_zero]),np.max(X[yaxis].values[idx_one])])
    
    im0 = ax0.scatter(
        X[xaxis].values[idx_zero], X[yaxis].values[idx_zero], 
        c = results_bag[idx_zero], linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'magma', vmin=0, vmax=1
    )
    plt.colorbar(im0)
    ax0.set_title('Labelled as Science (# = '+str(len(idx_zero))+')')
    ax0.set_xlim([xmin, xmax])
    ax0.set_ylim([ymin, ymax])
    ax0.set_xlabel(xaxis)
    ax0.set_ylabel(yaxis)
    
    ax1 = fig.add_subplot(122)
    im1 = ax1.scatter(X[xaxis].values[idx_one], X[yaxis].values[idx_one], 
        c = results_bag[idx_one], linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'magma', vmin=0, vmax=1)
    plt.colorbar(im1)
    ax1.set_title('Labelled as Background (# = '+str(len(idx_one))+')')
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    
    plt.tight_layout()
    plt.show()
    
def scipy_plot(X, y, results_bag, xaxis, yaxis, idx_zero, idx_one, thresh):
    # Makes density of the probabilities as a function of various spaces
    
    xmin = np.min([np.min(X[xaxis].values[idx_zero]),np.min(X[xaxis].values[idx_one])])
    xmax = np.max([np.max(X[xaxis].values[idx_zero]),np.max(X[xaxis].values[idx_one])])
    
    ymin = np.min([np.min(X[yaxis].values[idx_zero]),np.min(X[yaxis].values[idx_one])])
    ymax = np.max([np.max(X[yaxis].values[idx_zero]),np.max(X[yaxis].values[idx_one])])
    
    xs_sci = X[xaxis].values[idx_zero]
    ys_sci = X[yaxis].values[idx_zero]
    
    xs_bg = X[xaxis].values[idx_one]
    ys_bg = X[yaxis].values[idx_one]
    
    
    plt.clf()
    sns.set_style('dark')
    fig=plt.figure(figsize=(12,5))
    #plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.1)


    ax1=fig.add_subplot(121)
    ax1.set_title('Labelled as science (# = '+str(len(idx_zero))+')', loc='right')
    
     
    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ys_sci, 
                                                                         xs_sci, results_bag[idx_zero],
                                                                         statistic='mean', bins=50)
      
    heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(ys_bg, 
                                                                                  xs_bg, 
                                                                                  results_bag[idx_one],
                                                                                  statistic='mean', bins=50)
     
    
    if thresh == 0:
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma', vmin=0, vmax=1,
                     extent=[ ymin, ymax,xmin, xmax], interpolation='None')
    else:
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma', vmin=0, vmax=thresh,
                     extent=[ ymin, ymax,xmin, xmax], interpolation='None')
      
    ax1.set_ylim(xmin, xmax)
    ax1.set_xlim(ymin, ymax)
    ax1.set_aspect((ymax-ymin)/(xmax-xmin))

   
     
    plt.colorbar(im1, fraction=0.046, label=r'$p$')
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    #ax1.set_aspect((ymax-ymin)/(xmax-xmin))

    ax2=fig.add_subplot(122)
    ax2.set_title('Labelled as background (# = '+str(len(idx_one))+')', loc='right')
    if thresh==0:
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='magma', vmin=0, vmax=1,extent=[ymin, ymax, xmin, xmax])
    else:
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=thresh, vmax=1,extent=[ymin, ymax, xmin, xmax])
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046, label=r'$p$')
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()
    plt.show()
    #plt.savefig('../LDA_figures/gini_m20_separate_'+str(run)+'.pdf',  bbox_inches = 'tight')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_difference_scipy(X, results_bag, xaxis, yaxis, idx_zero, idx_one, thresh, bins, plot_kde):
    # Makes density of the probabilities as a function of various spaces
    
    xmin = np.min([np.min(X.iloc[idx_zero][xaxis]),np.min(X.iloc[idx_one][xaxis])])
    xmax = np.max([np.max(X.iloc[idx_zero][xaxis]),np.max(X.iloc[idx_one][xaxis])])
    
    ymin = np.min([np.min(X.iloc[idx_zero][yaxis]),np.min(X.iloc[idx_one][yaxis])])
    ymax = np.max([np.max(X.iloc[idx_zero][yaxis]),np.max(X.iloc[idx_zero][yaxis])])
    
    xs_sci = X.iloc[idx_zero][xaxis]
    ys_sci = X.iloc[idx_zero][yaxis]
    
    xs_bg = X.iloc[idx_one][xaxis]
    ys_bg = X.iloc[idx_one][yaxis]
    
    all_xs = X[xaxis]
    all_ys = X[yaxis]
    
    
    '''plt.clf()
    sns.set_style('dark')
    fig=plt.figure(figsize=(12,5))
    #plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.1)


    ax1=fig.add_subplot(121)
    ax1.set_title('Labelled as event (# = '+str(len(idx_zero))+')', loc='right')
    
     
    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ys_sci, 
                                                                         xs_sci, results_bag[idx_zero],
                                                                         statistic='mean', bins=50)'''
    
    heatmapall, xedgesall, yedgesall, binnumber = scipy.stats.binned_statistic_2d(all_ys, 
                                                                         all_xs, results_bag,
                                                                         statistic='mean', bins=bins)
      
    heatmapnon, xedgesnon, yedgesnon, binnumber = scipy.stats.binned_statistic_2d(ys_bg, 
                                                                                  xs_bg, 
                                                                                  results_bag[idx_one],
                                                                                  statistic='mean', bins=50)
    ''' 
    
    if thresh == 0:
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma',
                     extent=[ ymin, ymax,xmin, xmax], interpolation='None')
    else:
        im1 = ax1.imshow(np.flipud(heatmap), cmap='magma', vmin=0, vmax=thresh,
                     extent=[ ymin, ymax,xmin, xmax], interpolation='None')
      
    ax1.set_ylim(xmin, xmax)
    ax1.set_xlim(ymin, ymax)
    ax1.set_aspect((ymax-ymin)/(xmax-xmin))

   
     
    plt.colorbar(im1, fraction=0.046, label=r'$p$')
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    #ax1.set_aspect((ymax-ymin)/(xmax-xmin))

    ax2=fig.add_subplot(122)
    ax2.set_title('Labelled as background (# = '+str(len(idx_one))+')', loc='right')
    if thresh==0:
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='magma',extent=[ymin, ymax, xmin, xmax])
    else:
        im2 = ax2.imshow(np.flipud(heatmapnon), cmap='viridis', vmin=thresh, vmax=1,extent=[ymin, ymax, xmin, xmax])
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046, label=r'$p$')
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()
    plt.show()
    '''
    # Make a plot that is the higher probabilities minus the smaller probabilities
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)
    ax2.set_title('All events', loc='right')
    
    
    orig_cmap = matplotlib.cm.coolwarm
    shifted_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=thresh, stop=1, name='shrunk')
    
    im2 = ax2.imshow(np.flipud(heatmapall), cmap=shifted_cmap, extent=[ymin, ymax, xmin, xmax])
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046, label=r'$p$')
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()

    plt.show()
    
    
    if plot_kde==True:
        plt.clf()
        fig = plt.figure()
        ax2=fig.add_subplot(111)
        ax2.set_title('All events', loc='right')
        im2 = ax2.imshow(np.flipud(heatmapall), cmap=shifted_cmap, extent=[ymin, ymax, xmin, xmax])
        ax2.set_ylim(xmin, xmax)
        ax2.set_xlim(ymin, ymax)
        plt.colorbar(im2, fraction=0.046, label=r'$p$')
        ax2.set_xlabel(xaxis)
        ax2.set_ylabel(yaxis)
        ax2.set_aspect((ymax-ymin)/(xmax-xmin))

        # also, overplot hte kde of the original density

        sns.kdeplot(data = X, x=xaxis, y=yaxis, color='black')#subsample_sci['x'], subsample_sci['y'])


        plt.tight_layout()
        plt.show()
        
def plot_completeness_scipy(X, results_bag, xaxis, yaxis, idx_zero, idx_one, thresh, bins):
    # Makes density of the probabilities as a function of various spaces
    
    xmin = np.min([np.min(X.iloc[idx_zero][xaxis]),np.min(X.iloc[idx_one][xaxis])])
    xmax = np.max([np.max(X.iloc[idx_zero][xaxis]),np.max(X.iloc[idx_one][xaxis])])
    
    ymin = np.min([np.min(X.iloc[idx_zero][yaxis]),np.min(X.iloc[idx_one][yaxis])])
    ymax = np.max([np.max(X.iloc[idx_zero][yaxis]),np.max(X.iloc[idx_zero][yaxis])])
    
    xs_sci = X.iloc[idx_zero][xaxis]
    ys_sci = X.iloc[idx_zero][yaxis]
    
    xs_bg = X.iloc[idx_one][xaxis]
    ys_bg = X.iloc[idx_one][yaxis]
    
    all_xs = X[xaxis]
    all_ys = X[yaxis]
    
    
    '''plt.clf()
    sns.set_style('dark')
    fig=plt.figure(figsize=(12,5))
    #plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.1)


    ax1=fig.add_subplot(121)
    ax1.set_title('Labelled as event (# = '+str(len(idx_zero))+')', loc='right')
    
     
    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ys_sci, 
                                                                         xs_sci, results_bag[idx_zero],
                                                                         statistic='mean', bins=50)'''
    
    heatmapall, xedgesall, yedgesall = np.histogram2d(all_ys, 
                                                                         all_xs,  bins=bins)
    
    heatmapsci, xedgessci, yedgessci = np.histogram2d(ys_sci, 
                                                                         xs_sci, bins=bins)
      
    heatmapbg, xedgebg, yedgesbg = np.histogram2d(ys_bg, 
                                                                                  xs_bg, bins=bins)
     
    # Make a plot that is the higher probabilities minus the smaller probabilities
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)
    
    orig_cmap = matplotlib.cm.coolwarm
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shrunk')
    
    
    im2 = ax2.imshow(np.flipud(heatmapsci - heatmapbg), 
                     cmap='RdBu_r', 
                     extent=[ymin, ymax, xmin, xmax], 
                     norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03), interpolation ='None')
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()
    plt.title('Science - Background')
    plt.show()
    
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)
    
    im2 = ax2.imshow(100*(np.flipud(heatmapsci - heatmapbg) / np.flipud(heatmapall)), 
                     extent=[ymin, ymax, xmin, xmax], vmin=-100, vmax=100, cmap='RdBu_r', interpolation ='None')
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()
    plt.title('Completeness (% of events recovered)')
    plt.show()
    
    
def plot_purity_scipy(X, results_bag, xaxis, yaxis, idx_zero, idx_one, thresh, bins):
    # Makes density of the probabilities as a function of various spaces
    
    xmin = np.min([np.min(X.iloc[idx_zero][xaxis]),np.min(X.iloc[idx_one][xaxis])])
    xmax = np.max([np.max(X.iloc[idx_zero][xaxis]),np.max(X.iloc[idx_one][xaxis])])
    
    ymin = np.min([np.min(X.iloc[idx_zero][yaxis]),np.min(X.iloc[idx_one][yaxis])])
    ymax = np.max([np.max(X.iloc[idx_zero][yaxis]),np.max(X.iloc[idx_zero][yaxis])])
    
    xs_sci = X.iloc[idx_zero][xaxis]
    ys_sci = X.iloc[idx_zero][yaxis]
    
    xs_bg = X.iloc[idx_one][xaxis]
    ys_bg = X.iloc[idx_one][yaxis]
    
    all_xs = X[xaxis]
    all_ys = X[yaxis]
    
    
    '''plt.clf()
    sns.set_style('dark')
    fig=plt.figure(figsize=(12,5))
    #plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.1)


    ax1=fig.add_subplot(121)
    ax1.set_title('Labelled as event (# = '+str(len(idx_zero))+')', loc='right')
    
     
    heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ys_sci, 
                                                                         xs_sci, results_bag[idx_zero],
                                                                         statistic='mean', bins=50)'''
    
    heatmapall, xedgesall, yedgesall = np.histogram2d(all_ys, all_xs,  bins=bins)
    
    heatmapsci, xedgessci, yedgessci = np.histogram2d(ys_sci, xs_sci, bins=bins)
      
    heatmapbg, xedgebg, yedgesbg = np.histogram2d(ys_bg, xs_bg, bins=bins)
     
    # Make a plot that is the higher probabilities minus the smaller probabilities
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)
    
    im2 = ax2.imshow(np.flipud(heatmapbg - heatmapsci), 
                     cmap='RdBu_r', 
                     extent=[ymin, ymax, xmin, xmax], 
                     norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03), interpolation ='None')
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()
    plt.title('Background - Science')
    plt.show()
    
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)
    
    im2 = ax2.imshow(100*(np.flipud(heatmapbg - heatmapsci) / np.flipud(heatmapall)), 
                     extent=[ymin, ymax, xmin, xmax], vmin=-100, vmax=100, cmap='RdBu_r', interpolation ='None')
    ax2.set_ylim(xmin, xmax)
    ax2.set_xlim(ymin, ymax)
    plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((ymax-ymin)/(xmax-xmin))

    plt.tight_layout()
    plt.title('Purity (% of background events identified as such)')
    plt.show()
    
    
    
       
    
    
# Now plot the difference between the two
def comparison_heatmap(df1, df2, xaxis, yaxis, title):

    xs_grant = df1[xaxis]
    ys_grant = df1[yaxis]

    xs_murray = df2[xaxis]
    ys_murray = df2[yaxis]

    heatmapmurray, xedgesmurray, yedgesmurray = np.histogram2d(ys_murray, xs_murray)#, bins=bins)

    heatmapgrant, xedgegrant, yedgesgrant = np.histogram2d(ys_grant, xs_grant)#, bins=bins)

    # Make a plot that is the higher probabilities minus the smaller probabilities
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)

    orig_cmap = matplotlib.cm.coolwarm
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shrunk')


    im2 = ax2.imshow(np.flipud(heatmapmurray - heatmapgrant), 
                     cmap='RdBu_r', 
                     extent=[yedgesmurray[0], yedgesmurray[-1], xedgesmurray[0], xedgesmurray[-1]], 
                     norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03), interpolation ='None')
    ax2.set_ylim(xedgesmurray[0], xedgesmurray[-1])
    ax2.set_xlim(yedgesmurray[0], yedgesmurray[-1])
    plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    ax2.set_aspect((yedgesmurray[-1]-yedgesmurray[0])/(xedgesmurray[-1]-xedgesmurray[0]))

    plt.tight_layout()
    plt.title(title)
    plt.show()


def investigate_masks(df, radius_cutout):
    print('len going in', len(df))
    print(df.columns)
    # Things I think I need mask:
    #'PI exceeding 255'
    
    
    '''
    'AV3 corrected for ringing',
       'AU3 corrected for ringing', 
       'Event impacted by prior event (piled up)',
       'Shifted event time', 
       'Event telemetered in NIL mode',
       'V axis not triggered', 
       'U axis not triggered',
       'V axis center blank event', 
       'U axis center blank event',
       'V axis width exceeded', 
       'U axis width exceeded', 
       'Shield PMT active',
       'Upper level discriminator not exceeded',
       'Lower level discriminator not exceeded', 
       'Event in bad region',
       'Amp total on V or U = 0', 
       'Incorrect V center', 
       'Incorrect U center',
       'PHA ratio test failed', 
       'Sum of 6 taps = 0', 
       'Grid ratio test failed',
       'ADC sum on V or U = 0', 
       'PI exceeding 255',
       'Event time tag is out of sequence', 
       'V amp flatness test failed',
       'U amp flatness test failed', 
       'V amp saturation test failed',
       'U amp saturation test failed'
    '''
    
    #
    mask_these_indices = np.where( (df['AV3 corrected for ringing'].values==True)
                                | (df['AU3 corrected for ringing'].values == True) 
                                  | (df['Event impacted by prior event (piled up)'].values == True)
                                  | (df['Shifted event time'].values == True)
                                  | (df['Event telemetered in NIL mode'].values == True)
                                  | (df['V axis not triggered'].values == True)
                                  | (df['U axis not triggered'].values == True)
                                  | (df['V axis center blank event'].values == True)
                                  | (df['U axis center blank event'].values == True)
                                  | (df['V axis width exceeded'].values == True)
                                  | (df['U axis width exceeded'].values == True)
                                  | (df['Shield PMT active'].values == True)
                                  | (df['Upper level discriminator not exceeded'].values == True)
                                  | (df['Lower level discriminator not exceeded'].values == True)
                                  | (df['Event in bad region'].values == True)
                                  | (df['Amp total on V or U = 0'].values == True)
                                  | (df['Incorrect V center'].values == True)
                                  | (df['Incorrect U center'].values == True)
                                  | (df['PHA ratio test failed'].values == True)
                                  | (df['Sum of 6 taps = 0'].values == True)
                                  | (df['Grid ratio test failed'].values == True)
                                  | (df['ADC sum on V or U = 0'].values == True)
                                  | (df['PI exceeding 255'].values == True)
                                  | (df['Event time tag is out of sequence'].values == True)
                                  | (df['V amp flatness test failed'].values == True)
                                  | (df['U amp flatness test failed'].values == True)
                                  | (df['V amp saturation test failed'].values == True)
                                  | (df['U amp saturation test failed'].values == True)
                                  )[0]#, df['PI exceeding 255'].values)
    dont_mask_these_indices = np.where( (df['AV3 corrected for ringing'].values==False)
                                & (df['AU3 corrected for ringing'].values == False) 
                                  & (df['Event impacted by prior event (piled up)'].values == False)
                                  & (df['Shifted event time'].values == False)
                                  & (df['Event telemetered in NIL mode'].values == False)
                                  & (df['V axis not triggered'].values == False)
                                  & (df['U axis not triggered'].values == False)
                                  & (df['V axis center blank event'].values == False)
                                  & (df['U axis center blank event'].values == False)
                                  & (df['V axis width exceeded'].values == False)
                                  & (df['U axis width exceeded'].values == False)
                                  & (df['Shield PMT active'].values == False)
                                  & (df['Upper level discriminator not exceeded'].values == False)
                                  & (df['Lower level discriminator not exceeded'].values == False)
                                  & (df['Event in bad region'].values == False)
                                  & (df['Amp total on V or U = 0'].values == False)
                                  & (df['Incorrect V center'].values == False)
                                  & (df['Incorrect U center'].values == False)
                                  & (df['PHA ratio test failed'].values == False)
                                  & (df['Sum of 6 taps = 0'].values == False)
                                  & (df['Grid ratio test failed'].values == False)
                                  & (df['ADC sum on V or U = 0'].values == False)
                                  & (df['PI exceeding 255'].values == False)
                                  & (df['Event time tag is out of sequence'].values == False)
                                  & (df['V amp flatness test failed'].values == False)
                                  & (df['U amp flatness test failed'].values == False)
                                  & (df['V amp saturation test failed'].values == False)
                                  & (df['U amp saturation test failed'].values == False)
                                  )[0]#, df['PI exceeding 255'].values)
    #print('indices mask', mask_these_indices, 'og length', len(df), 'length masked', len(mask_these_indices))
    #print('inverse mask', dont_mask_these_indices, len(dont_mask_these_indices))
    
    
    '''mask_these_indices = []
    
    for j in range(len(df)):
        if row[j]==True:
            #print(df.values[j])
            mask_these_indices.append(j)
    '''        
    # Okay figure out what these events look like in this space:
    
    df_mask = pd.DataFrame(df.values[mask_these_indices], columns = df.columns)
    df_clean = pd.DataFrame(df.values[dont_mask_these_indices], columns = df.columns)
    print('length masked', len(df_mask))
    print('length clean', len(df) - len(df_mask), len(df_clean))
    # Okay, put these back into the dataframes
    
    
    
    '''
    Let's try to actually get the hyperbola drawn on here then
    hyperbola = b * np.sqrt(((fb - h)**2 / a**2) - 1)

        return hyperbola

    def legacy_hyperbola_test(self, tolerance=0.035):
        """[summary]
        Keyword Arguments:
            tolerance {float} -- [description] (default: {0.035})
        Returns:
            [type] -- [description]
        """

        # Remind the user what tolerance they're using
        # print("{0: <25}| Using tolerance = {1}".format(" ", tolerance))

        # Set hyperbolic coefficients, depending on whether this is HRC-I or -S
        if self.detector == "HRC-I":
            a_u = 0.3110
            b_u = 0.3030
            h_u = 1.0580

            a_v = 0.3050
            b_v = 0.2730
            h_v = 1.1
    '''
    
    fbs = np.linspace(0,1,1000)
    
    a = 0.311
    b = 0.303
    h = 1.0580
    
    fps_lower = [b * np.sqrt(((x - h - 0.03)**2 / a**2) - 1) for x in fbs]
    fps_upper = [b * np.sqrt(((x - h + 0.03)**2 / a**2) - 1) for x in fbs]
    
    fps_lower_lower = [-b * np.sqrt(((x - h - 0.03)**2 / a**2) - 1) for x in fbs]
    fps_upper_lower = [-b * np.sqrt(((x - h + 0.03)**2 / a**2) - 1) for x in fbs]
    
    plt.clf()
    
    xs = df_mask['fb_u'].values
    ys = df_mask['fp_u'].values
    

    plt.scatter(
        xs, ys, color='red',linewidth = 0, s = 5, alpha = 0.95, 
         vmax=1, vmin=0, label='Mask'
    )
    xs = df_clean['fb_u'].values
    ys = df_clean['fp_u'].values
    

    plt.scatter(
        xs, ys, color='black',linewidth = 0, s = 5, alpha = 0.95, 
         vmax=1, vmin=0, label='Clean'
    )
    
    
    
    plt.plot(fbs, fps_lower, color='black')
    plt.plot(fbs, fps_upper, color='black')
    
    plt.plot(fbs, fps_lower_lower, color='black')
    plt.plot(fbs, fps_upper_lower, color='black')
    plt.legend()
    
    plt.show()
    
    plt.clf()
    
    xs = df_mask['pha'].values
    ys = df_mask['sumamps'].values
    

    plt.scatter(
        xs, ys, color='red',linewidth = 0, s = 5, alpha = 0.95, 
         vmax=1, vmin=0
    )
    
    xs = df_clean['pha'].values
    ys = df_clean['sumamps'].values
    

    plt.scatter(
        xs, ys, color='black',linewidth = 0, s = 5, alpha = 0.95, 
         vmax=1, vmin=0
    )
    
    
    plt.show()
    
    
    
    
    '''Make beautiful plots of the cut events'''
    
    
    mean_x = np.mean(df_clean.x)
    mean_y = np.mean(df_clean.y)
    
    print(df_clean.x)
    #print([np.sqrt((df_clean.x- mean_x)**2 + (df_clean.y - mean_y)**2) < radius_cutout])
    
    x = df_clean['x']#[np.sqrt((df_clean['x'].values- mean_x)**2 + (df_clean['y'].values - mean_y)**2) < radius_cutout]
    y = df_clean['y']#[np.sqrt((df_clean['x'] - mean_x)**2 + (df_clean['y'] - mean_y)**2) < radius_cutout]





    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    sns.set_style('dark')
    plt.clf()
    plt.imshow(ma.masked_where(img_data==0, img_data),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        plt.colorbar()
    except:
        print('no colorbar')
    plt.title(' Accepted Mask')
    plt.show()
    
    x = df_mask['x']#[np.sqrt((df_mask.x - mean_x)**2 + (df_mask.y - mean_y)**2) < radius_cutout]
    y = df_mask['y']#[np.sqrt((df_mask.x - mean_x)**2 + (df_mask.y - mean_y)**2) < radius_cutout]





    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    sns.set_style('dark')
    plt.clf()
    plt.imshow(ma.masked_where(img_data==0, img_data),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        plt.colorbar()
    except:
        print('no colorbar')
    plt.title(' Dirty Mask')
    plt.show()
    
    return df_clean, df_mask
            
def classify_sci_cutout(science_obsid_train, bg_obsid_train,
                                            science_obsid_test, bg_obsid_test,
                                            sci_label, 
                                            background_label, 
                                            radius_cutout,
                                            nbins,
                                            subsamplesize,
                                            NP,# number of positives for bagging
                                            th,
                                            bagging_hyperparameters,       
                                            predictor_list,# this is for the bag
                                            feature_list_RFR,
                                            plot_data = False,
                                            run_hyperbola = True,
                                            run_forest = True,
                                            balanced = True,
                                            evt2_filter = False,
                                            adjust_th = True):
    
    
    
    

    file_dir = "/Users/beckynevin/CfA_Code/HRC/hyperscreen/notebooks/csv_files/"
    

    obs = hypercore_csv.HRCevt1(file_dir+'science_dataframe_'+str(science_obsid_test)+'.csv')
    
    
    
    
    # Okay step one is to see if this output has anything that should me excluded going
    # into event2 files
    if evt2_filter:
        data_clean, data_dirty = investigate_masks(obs.data.dropna().sample(n=1000), radius_cutout)#.sample(n=1000)
    else:
        data_clean = obs.data.dropna()
    
    
    bg_train = hypercore_csv.HRCevt1(file_dir+'science_dataframe_cutout_'+str(bg_obsid_train)+'.csv')
    bg_test = hypercore_csv.HRCevt1(file_dir+'science_dataframe_cutout_'+str(bg_obsid_test)+'.csv')
    
    
    
    if evt2_filter:
        print('EVT2 filter')
        data_bg_train, data_bg_train_dirty = investigate_masks(bg_train.data.dropna().sample(n=1000), radius_cutout)
        data_bg_test, data_bg_train_dirty = investigate_masks(bg_train.data.dropna().sample(n=1000), radius_cutout)
    else:
        data_bg_train = bg_train.data.dropna()
        data_bg_test = bg_test.data.dropna()
        
    
    
    
    
    
    if run_hyperbola:
        # Test 1: Murray~~~~~

        ## Now plot what happenes with the Murray classification:
        mask_steve = data_clean['Hyperbola test failed'].values
        print('~~~~~~~~~Results from Murray~~~~~~~')
        
        print('mask_steve', type(mask_steve), mask_steve)
        print('~mask_steve', np.invert(mask_steve), ~mask_steve)#, ~list(mask_steve))
        

        df_all = data_clean.dropna()
        df = data_clean[np.invert(mask_steve)].dropna()
        # Okay so these are accepted, zeros
        labels_steve = []
        
        
        df_dropped = data_clean[mask_steve].dropna()
        for k in range(len(df_dropped)):
            labels_steve.append(1)
        
        df_all = pd.concat([ df_dropped, df])
        
        # Def make sure to put the reals on top
        for k in range(len(df)):
            labels_steve.append(0)#0 for x in df]
        
        

        # Then do the cutout selection

        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_all_select = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]
        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]

        print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        sns.set_style('dark')
        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Murray')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        sns.set_style('dark')
        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Rejected Murray')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
        
        
        
        #~~~~~~~~~~~~~~~~~~
        # BUT YOU NEED TO MAKE AN ARRAY THAT IS THE MURRAY CLASSIFICATION RESULT
        xs = df_all['fb_u'].values
        ys = df_all['fp_u'].values
        cs = labels_steve

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['pha'].values
        ys = df_all['sumamps'].values
        cs = labels_steve

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['x'].values
        ys = df_all['y'].values
        cs = labels_steve

        '''plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()'''
        
        
        
        

        #~~~~~~~~~~~~~~~~~~~~~

        ## Now, do the same thing but for the background




        # Test 1: Murray~~~~~

        ## Now plot what happenes with the Murray classification:
        mask_steve = data_bg_test['Hyperbola test failed']
        print('~~~~~~~~~Results from Murray~~~~~~~')


        df_all = data_bg_test
        df = data_bg_test[~mask_steve].dropna()
        df_dropped = data_bg_test[mask_steve].dropna()
        
       
        
        
        
        
        

        # Then do the cutout selection

        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_all_select = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]
        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]

        print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]






        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Accepted Murray')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]






        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Rejected Murray')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()


        # Test 2: Grant~~~~~~
        ## Now run hyperscreen
        print('~~~~~~~~~~~~~~~~~~Hyperscreen~~~~~~~~~~~~~~~~~~~')
        tremblay = obs.hyperscreen()

        mask_grant = tremblay['All Survivals (boolean mask)']


        df = obs.data[mask_grant].dropna()
        df_all = obs.data.dropna()
        df_dropped = obs.data[~mask_grant].dropna()
        
        df_all = pd.concat([df_dropped, df])
        
        labels_grant = []
        for k in range(len(df_dropped)):
            labels_grant.append(1)
        for k in range(len(df)):
            labels_grant.append(0)


        # Then do the cutout selection
        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        df_select_all = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        print('fraction retrieved', len(df_select)/len(df_select_all))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]




        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Grant')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]




        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Rejected Grant')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
        
        #~~~~~~~~~~~~~~~~~~
        # BUT YOU NEED TO MAKE AN ARRAY THAT IS THE MURRAY CLASSIFICATION RESULT
        xs = df_all['fb_u'].values
        ys = df_all['fp_u'].values
        cs = labels_grant

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['pha'].values
        ys = df_all['sumamps'].values
        cs = labels_grant

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['x'].values
        ys = df_all['y'].values
        cs = labels_grant

        '''plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()'''
        

        #~~~~~~~~~~~~~~~~~~~~~


        # Test 2: Grant~~~~~~
        ## Now run hyperscreen
        print('~~~~~~~~~~~~~~~~~~Hyperscreen~~~~~~~~~~~~~~~~~~~')
        tremblay = bg_test.hyperscreen()

        mask_grant = tremblay['All Survivals (boolean mask)']


        df = bg_test.data[mask_grant].dropna()
        df_all = bg_test.data.dropna()
        df_dropped = bg_test.data[~mask_grant].dropna()

        # Then do the cutout selection
        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        df_select_all = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        print('fraction retrieved', len(df_select)/len(df_select_all))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]



        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Accepted Grant')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]



        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Rejected Grant')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()
    
    
    
    # Test 3: Random Forest Baseline~~~~~~~~
    
    result_RFR, result_bag, sci_train, bg_train = prep_data_ML_sci(science_obsid_train, bg_obsid_train, subsamplesize, NP, balanced=balanced)
    
    # This is for testing
    result_test, result_test_bag, sci_test, bg_test = prep_data_ML_sci(science_obsid_test, bg_obsid_test, subsamplesize, NP, balanced=balanced)
    # Cut down sci and bg because they are too big!
    bg = bg_test.sample(n=NP, random_state=1)
    sci = sci_test.sample(n=subsamplesize, random_state=1)
    
    

    
    
    random_numbers = np.random.random(size=len(result_RFR))

    result_RFR['random'] = random_numbers
    
    random_numbers = np.random.random(size=len(result_bag))

    result_bag['random'] = random_numbers
    
    random_numbers = np.random.random(size=len(sci_test))

    sci_test['random'] = random_numbers

    random_numbers = np.random.random(size=len(bg_test))

    bg_test['random'] = random_numbers
    
    # Now apply the classification to the entire cutout:
    classify_sci = sci_test.dropna()
    classify_bg = bg_test.dropna()
    
    if plot_data:
        #science_obsid_train, bg_obsid_train,science_obsid_test, bg_obsid_test
        nbins=500
        x = sci_train['x']
        y = sci_train['y']
        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xinterval = (xedges[-1] - xedges[0])/nbins
        yinterval = (yedges[-1] - yedges[0])/nbins
        mask = ma.masked_where(img_data==0, img_data)

        plt.clf()
        fig = plt.figure()
        ax0 = fig.add_subplot(221)
        im0 = ax0.imshow(img_data, rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.colorbar(im0, fraction=0.046)
        ax0.set_title('Training Science '+str(science_obsid_train))
        ax0.axis('off')
        
        x = bg_train['x']
        y = bg_train['y']
        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xinterval = (xedges[-1] - xedges[0])/nbins
        yinterval = (yedges[-1] - yedges[0])/nbins
        mask = ma.masked_where(img_data==0, img_data)
        
        ax1 = fig.add_subplot(222)
        im1 = ax1.imshow(img_data, rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.colorbar(im1, fraction=0.046)
        ax1.set_title('Training BG '+str(bg_obsid_train))
        ax1.axis('off')
        
        x = classify_sci['x']
        y = classify_sci['y']
        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xinterval = (xedges[-1] - xedges[0])/nbins
        yinterval = (yedges[-1] - yedges[0])/nbins
        mask = ma.masked_where(img_data==0, img_data)
        
        ax2 = fig.add_subplot(223)
        im2 = ax2.imshow(img_data, rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.colorbar(im1, fraction=0.046)
        ax2.set_title('Test Science '+str(science_obsid_test))
        ax2.axis('off')
        
        x = classify_bg['x']
        y = classify_bg['y']
        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xinterval = (xedges[-1] - xedges[0])/nbins
        yinterval = (yedges[-1] - yedges[0])/nbins
        mask = ma.masked_where(img_data==0, img_data)
        
        ax3 = fig.add_subplot(224)
        im3 = ax3.imshow(img_data, rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.colorbar(im3, fraction=0.046)
        ax3.set_title('Test BG '+str(bg_obsid_test))
        ax3.axis('off')
        
        
        plt.show()
    
    
    if run_forest:
        print('~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~')
        #['fp_u', 'fb_u', 'fp_v', 'fb_v','pha', 'sumamps', 'random']
        X = result_RFR[feature_list_RFR]



        terms_RFR, reject_terms_RFR, model_RFR = run_RFR(result_RFR, feature_list_RFR, 'no')




        preds_sci_RFR = model_RFR.predict(classify_sci[feature_list_RFR])
        print('sci rfr', preds_sci_RFR)



        # Now make the plot

        # < 0.5 is sci
        ind_select = np.where(np.array(preds_sci_RFR) < th)[0]
        ind_select_drop = np.where(np.array(preds_sci_RFR) > th)[0]

        df = classify_sci.iloc[ind_select]
        df_dropped = classify_sci.iloc[ind_select_drop]
        print('len of selected', len(df), 'len rejected', len(df_dropped))
        print('len before selected', len(classify_sci))


        df_all = classify_sci[np.sqrt((classify_sci['x'] - np.mean(classify_sci['x']))**2 + (classify_sci['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(classify_sci['x']))**2 + (df['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        #print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        print('fraction retrieved', len(df_select)/len(df_all))
        print('len of each of these', len(df_select), len(df_all))

        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Random Forest')
        plt.show()

        x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
        except:
            print('len of xs', len(x))
        plt.title(sci_label+' Rejected Random Forest')
        plt.show()
    
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
    
        # Test 3: Baseline random forest~~~~~~~~
        print('~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~')
        # > 0.5 is background
        preds_bg_RFR = model_RFR.predict(classify_bg[feature_list_RFR])

        ind_bg = np.where(np.array(preds_bg_RFR) < th)[0]
        ind_bg_drop = np.where(np.array(preds_bg_RFR) > th)[0]


        df = classify_bg.iloc[ind_bg]
        df_dropped = classify_bg.iloc[ind_bg_drop]




        df_all = classify_bg[np.sqrt((classify_bg['x'] - np.mean(classify_bg['x']))**2 + (classify_bg['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(classify_bg['x']))**2 + (df['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        #print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        print('fraction retrieved', len(df_select)/len(df_all))
        print('lens', len(df_select), len(df_all))

        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
            plt.title(background_label+' Accepted Random Forest')
            plt.show()
        except:
            print('nothing here', len(y))

        x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
            plt.title(background_label+' Rejected Random Forest')
            plt.show()
        except:
            print('nothing here', len(y))
            
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()
    
    
    # Test 4: PUbag
    print('~~~~~~Bagging PU~~~~~~~')
    feature_list_bag = predictor_list#['fp_u', 'fb_u', 'fp_v', 'fb_v','pha', 'sumamps']
    X = result_bag[feature_list_bag]
    y = result_bag['class'].values
    
    n_estimators, max_samples, len_features = bagging_hyperparameters#bagging_hyperparameters
    
    if max_samples=='balance':# this means you want to do balanced sampling
        max_samples = sum(y>0)
    
    print('length of background examples', sum(y>0), 'length of science', len(y) - sum(y>0), 'total length y',len(y))
    print('# of decision trees', n_estimators)
    print('# of unlabeled samples to draw for each', max_samples)
    print('# of features', len_features)
    
    bc = BaggingPuClassifier(base_estimator=None, 
                             n_estimators=n_estimators,# this is the number of decision trees (SVMs) 
                             max_samples=max_samples, # this is the number of unlabeled samples or K to draw
                             max_features=len_features, 
                             bootstrap=True, 
                             bootstrap_features=False, 
                             oob_score=True, 
                             warm_start=False, 
                             n_jobs=-1, 
                             random_state=True, 
                             verbose=1)


    bc.fit(X, y)
    

    # Store the scores assigned by this approach
    results_bag = pd.DataFrame({
        'label'      : y},        # The labels to be shown to models in experiment
        columns = ['label'])
    results_bag['output_bag'] = bc.oob_decision_function_[:,1]


    # Now use it to predict the test case
    pred_sci_bag = bc.predict_proba(classify_sci[feature_list_bag])[:,1]
    pred_bg_bag = bc.predict_proba(classify_bg[feature_list_bag])[:,1]
    
    if adjust_th:
        #then th will be equal to something else
        
        th = compute_thresh(classify_sci, classify_bg, pred_sci_bag, pred_bg_bag, radius_cutout)
        print('new thresh', th)
        
 
        
    
    
    #~~~~~~~~~~~~~~~~~~~`~~~~~~~~~~~~
    # Now that I have an adjusted threshold, I wonder if its possible to set the midpoint
    xs = classify_sci['fb_u'].values
    ys = classify_sci['fp_u'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,u}$')
    plt.ylabel(r'$f_{p,u}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    xs = classify_sci['fb_v'].values
    ys = classify_sci['fp_v'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    
    xs = classify_sci['pha'].values
    ys = classify_sci['sumamps'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$PHA$')
    plt.ylabel(r'$SUMAMPS$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    xs = classify_sci['x'].values
    ys = classify_sci['y'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    

    indices_sci = np.where(np.array(pred_sci_bag) < th)[0]
    indices_sci_drop = np.where(np.array(pred_sci_bag) > th)[0]
    df = classify_sci.iloc[indices_sci]
    df_dropped = classify_sci.iloc[indices_sci_drop]
    
    
    df_all = classify_sci[np.sqrt((classify_sci['x'] - np.mean(classify_sci['x']))**2 + (classify_sci['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
    df_select = df[np.sqrt((df['x'] - np.mean(classify_sci['x']))**2 + (df['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
    
    #print('same thing but for cutout selection',
    #      'len all', len(df_all_select), 'len selected', len(df_select))
    #print('fraction retrieved', len(df_select)/len(df_all_select))
    
    x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    print('fraction retrieved', len(df_select)/len(df_all))
    print('len', len(df_select), len(df_all))
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data==0, img_data),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Accepted PUBagging')
    plt.show()
    
    x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        plt.colorbar()
    except:
        print(len(x))
    plt.title(sci_label+' Rejected PUBagging')
    plt.show()
    
    plt.clf()
    plt.imshow(img_data/img_data_rej,  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Ratio of accepted to rejected')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Now use it to predict the test case
    pred_bg_bag = bc.predict_proba(classify_bg[feature_list_bag])[:,1]
    
    
    indices_bg_bag = np.where(np.array(pred_bg_bag) < th)
    indices_bg_bag_drop = np.where(np.array(pred_bg_bag) > th)
    df_bag = classify_bg.iloc[indices_bg_bag]
    df_dropped = classify_bg.iloc[indices_bg_bag_drop]
    
    
    #~~~~~~~~~~~~~~~~~~
    
    xs = classify_bg['fb_u'].values
    ys = classify_bg['fp_u'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,u}$')
    plt.ylabel(r'$f_{p,u}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    xs = classify_bg['fb_v'].values
    ys = classify_bg['fp_v'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    xs = classify_bg['pha'].values
    ys = classify_bg['sumamps'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$PHA$')
    plt.ylabel(r'$SUMAMPS$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    '''xs = classify_bg['x'].values
    ys = classify_bg['y'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'magma', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()'''
    
    
    #~~~~~~~~~~~~~~~~~~~~~
    
    df_all_bag = classify_bg[np.sqrt((classify_bg['x'] - np.mean(classify_bg['x']))**2 + (classify_bg['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
    df_select_bag = df_bag[np.sqrt((df_bag['x'] - np.mean(classify_bg['x']))**2 + (df_bag['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
    
    #print('same thing but for cutout selection',
    #      'len all', len(df_all_select), 'len selected', len(df_select))
    #print('fraction retrieved', len(df_select)/len(df_all_select))
    
    x_bag = df_bag['x'][np.sqrt((df_bag['x'] - np.mean(df_bag['x']))**2 + (df_bag['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    y_bag = df_bag['y'][np.sqrt((df_bag['x'] - np.mean(df_bag['x']))**2 + (df_bag['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_bag, yedges_bag, xedges_bag = np.histogram2d(y_bag, x_bag, nbins)
    extent = [xedges_bag[0], xedges_bag[-1], yedges_bag[0], yedges_bag[-1]]

    print('fraction retrieved', len(df_select_bag)/len(df_all_bag))
    print(len(df_select_bag), len(df_all_bag))
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data_bag==0, img_data_bag),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.colorbar()
    plt.title(sci_label+' Accepted PUBagging')
    plt.show()
    
    x_bag = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_bag['x']))**2 + (df_dropped['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    y_bag = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_bag['x']))**2 + (df_dropped['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_bag_rej, yedges_bag, xedges_bag = np.histogram2d(y_bag, x_bag, nbins)
    extent = [xedges_bag[0], xedges_bag[-1], yedges_bag[0], yedges_bag[-1]]

    
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data_bag_rej==0, img_data_bag_rej),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.colorbar()
    plt.title(sci_label+' Rejected PUBagging')
    plt.show()
    
    plt.clf()
    plt.imshow(img_data_bag/img_data_bag_rej,  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Ratio of accepted to rejected')
    plt.show()
    
    if run_forest ==True:
    
        plt.clf()
        plt.hist(preds_sci_RFR, label='Science probabilities', alpha=0.5, bins=30)
        plt.hist(preds_bg_RFR, label='Background probabilities', alpha=0.5, bins=30)
        plt.legend()
        plt.title('RFR')
        plt.show()
    
    plt.clf()
    plt.hist(pred_sci_bag, label='Science probabilities', alpha=0.5, bins=30)
    plt.hist(pred_bg_bag, label='Background probabilities', alpha=0.5, bins=30)
    plt.axvline(x=th)
    plt.legend()
    plt.title('PU Bagging')
    plt.show()
    if run_forest:
        return preds_sci_RFR, preds_bg_RFR, pred_sci_bag, pred_bg_bag, classify_sci, classify_bg
    else:
        return 0, 0, pred_sci_bag, pred_bg_bag, classify_sci, classify_bg
        


def prep_data_ML_stowed(obsid, stowed, n_points, NP, balanced=True):
    file_dir = "/Users/beckynevin/CfA_Code/HRC/hyperscreen/notebooks/csv_files/"
    try:
        sci = pd.read_csv(file_dir+'science_dataframe_'+str(obsid)+'.csv', index_col=0)
    except:
        sci = obsid
    # Make a dataframe out of a stowed file
    bg = stowed#pd.read_csv(file_dir+'bg_dataframe_'+str(obsid_bg)+'.csv', index_col=0)
    
    
    # Right of. thebat make sure you don't need to drop stuff
    try:
        bg = bg.drop(columns=['time','samp'])
    except:
        try:
            bg = bg.drop(columns='time')
        except:
            print('no columns')
            
    try:
        sci = sci.drop(columns=['time','samp'])
    except:
        try:
            sci = sci.drop(columns='time')
        except:
            print('no columns')
    
    # First, get rid of the columns that have only NaNs:
    
    
    a_u = bg["au1"]  # otherwise known as "a1"
    b_u = bg["au2"]  # "a2"
    c_u = bg["au3"]  # "a3"

    a_v = bg["av1"]
    b_v = bg["av2"]
    c_v = bg["av3"]

    with np.errstate(invalid='ignore'):
        # Do the U axis
        fp_u = ((c_u - a_u) / (a_u + b_u + c_u))
        fb_u = b_u / (a_u + b_u + c_u)

        # Do the V axis
        fp_v = ((c_v - a_v) / (a_v + b_v + c_v))
        fb_v = b_v / (a_v + b_v + c_v)

    bg['fp_u'] = fp_u
    bg['fb_u'] = fb_u
    bg['fp_v'] = fp_v
    bg['fb_v'] = fb_v
    
    a_u = sci["au1"]  # otherwise known as "a1"
    b_u = sci["au2"]  # "a2"
    c_u = sci["au3"]  # "a3"

    a_v = sci["av1"]
    b_v = sci["av2"]
    c_v = sci["av3"]

    with np.errstate(invalid='ignore'):
        # Do the U axis
        fp_u = ((c_u - a_u) / (a_u + b_u + c_u))
        fb_u = b_u / (a_u + b_u + c_u)

        # Do the V axis
        fp_v = ((c_v - a_v) / (a_v + b_v + c_v))
        fb_v = b_v / (a_v + b_v + c_v)

    sci['fp_u'] = fp_u
    sci['fb_u'] = fb_u
    sci['fp_v'] = fp_v
    sci['fb_v'] = fb_v
    
    
    
     
    
    # Now create class labels, label all of the sci events as 'unlabelled' or '0'
    # and all of the background events as 'positive' or '1'
    bg['class'] = 1
    sci['class'] = 0
    
    
    '''
    # Now, I need to write a code that can cut out the brightest stars in the 'background' so they
    # are not included in the training set
    
    bg = cutout_brightpatches(bg, 5000)
    
    STOP
    '''
    
    #print('sci', sci.columns)
    #print('bg', bg.columns)
    
    
    
    # print('length of positives', len(bg), 'length of unlabelled', len(sci))
    # now randomly sample from each
    #n_points = int(1e5)
    if balanced==False: #this means we are putting together the training set for the RFR where we want the sample sizes to be equal
        frames_bag = [bg.sample(n=NP, random_state=1), sci.sample(n=n_points, random_state=1)]
        frames_RFR = [bg.sample(n=int(n_points/5), random_state=1), sci.sample(n=n_points, random_state=1)]
        
        result_bag = pd.concat(frames_bag)
        #drop nans:
        #result_bag = result_bag.drop(columns='time')
        result_bag = result_bag.dropna()

        result_RFR = pd.concat(frames_RFR)
        #drop nans:
        #result_RFR = result_RFR.drop(columns='time')
        result_RFR = result_RFR.dropna()
        
        #print('the total length should be ', NP, n_points, NP+n_points)
        #print(result_bag.columns)

        return result_RFR, result_bag, sci, bg # result_bag is NP of background and subsamplesize of sci
    else:
        frames = [bg.sample(n=n_points, random_state=1), sci.sample(n=n_points, random_state=1)]
        #frames = [bg, sci]
        result = pd.concat(frames)
        #drop nans:
        #result = result.drop(columns='time')
        result = result.dropna()
        return result, result, sci, bg
        
    '''
    else:
    
        try:
            frames = [bg.sample(n=n_points, random_state=1), sci]
        except ValueError:
            STOP
            # This means n_points is bigger than the dataframe
            n_points = int(1e4)
            try:
                frames = [bg.sample(n=n_points, random_state=1), sci.sample(n=n_points, random_state=1)]
            except:
                n_points = int(1e3)
                try:
                    frames = [bg.sample(n=n_points, random_state=1), sci.sample(n=n_points, random_state=1)]
                except:
                    return 0
    '''
def compute_thresh(sci, bg, pred_sci, pred_bg, radius_cutout):
    print('~~~~~~Adjusting Threshold~~~~~~')
    threshold_list = np.linspace(0.1,0.9,9)
    
    p_sci = []
    p_bg = []
    for j in range(len(threshold_list)):
        
        th = threshold_list[j]
        indices_sci = np.where(np.array(pred_sci) < th)[0]
        df = sci.iloc[indices_sci]


        df_all = sci[np.sqrt((sci['x'] - np.mean(sci['x']))**2 + (sci['y'] - np.mean(sci['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(sci['x']))**2 + (df['y'] - np.mean(sci['y']))**2) < radius_cutout]

        try:
            #per_sci = len(indices_sci)/len(pred_sci)#df_select/df_all
            pe_sci = len(df_select)/len(df_all)#df_select/df_all
        except:
            #per_sci = 0
            pe_sci = 0
        indices_bg_bag = np.where(np.array(pred_bg) < th)[0]
        df_bag = bg.iloc[indices_bg_bag]

        df_all_bag = bg[np.sqrt((bg['x'] - np.mean(sci['x']))**2 + (bg['y'] - np.mean(sci['y']))**2) < radius_cutout]
        df_select_bag = df_bag[np.sqrt((df_bag['x'] - np.mean(sci['x']))**2 + (df_bag['y'] - np.mean(sci['y']))**2) < radius_cutout]
        #print('percent of sci maintained', df_select/df_all, 'percent of bg kept', df_select_bag/df_all_bag)
        try:
            #per_bg = len(indices_bg_bag)/len(pred_bg)#df_select_bag/df_all_bag
            pe_bg = len(df_select_bag)/len(df_all_bag)
        except:
            #per_bg = 1
            pe_bg = 1
            
        '''plt.clf()
        plt.hist(pred_sci, label='Science probabilities', alpha=0.5, bins=30)
        plt.hist(pred_bg, label='Background probabilities', alpha=0.5, bins=30)
        plt.legend()
        #plt.annotate('Percent Sci = '+str(round(per_sci,2)), xy=(0.02,0.1), xycoords='axes fraction' )
        #plt.annotate('Percent BG = '+str(round(per_bg,2)), xy=(0.02,0.05), xycoords='axes fraction' )
        
        plt.annotate('Percent Sci = '+str(round(pe_sci,2)), xy=(0.02,0.3), xycoords='axes fraction' )
        plt.annotate('Percent BG = '+str(round(pe_bg,2)), xy=(0.02,0.25), xycoords='axes fraction' )
        
        
        plt.axvline(x=th)
        plt.show()'''
        
        p_sci.append(pe_sci)
        p_bg.append(pe_bg)
        
    diff = np.array(p_sci)-np.array(p_bg)
    plt.clf()
    plt.plot(threshold_list, p_sci, label='Percent of science kept')
    plt.plot(threshold_list, p_bg, label='Percent of background kept')
    #plt.plot(threshold_list, diff, label='Difference')
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(threshold_list, diff, label='Difference')
    plt.legend()
    plt.show()
    
    
    idx = list(diff).index(np.max(diff))
    
    return threshold_list[idx]



def test_comparison_sci_sci(science_obsid_train, bg_obsid_train,
                                            science_obsid_test, bg_obsid_test,
                                            sci_label, 
                                            background_label, 
                                            radius_cutout,
                                            nbins,
                                            subsamplesize,
                                            NP,# number of positives for bagging
                                            th,
                                            bagging_hyperparameters,       
                                            predictor_list,# this is for the bag
                                            feature_list_RFR,
                                            run_hyperbola = True,
                                            run_forest = True,
                                            balanced = True,
                                            evt2_filter = False,
                                            adjust_th = True):
    
    
    
    

    file_dir = "/Users/beckynevin/CfA_Code/HRC/hyperscreen/notebooks/csv_files/"
    

    obs = hypercore_csv.HRCevt1(file_dir+'science_dataframe_'+str(science_obsid_test)+'.csv')
    
    
    
    
    # Okay step one is to see if this output has anything that should me excluded going
    # into event2 files
    if evt2_filter:
        data_clean, data_dirty = investigate_masks(obs.data.dropna().sample(n=1000), radius_cutout)#.sample(n=1000)
    else:
        data_clean = obs.data.dropna()
    
    
    bg_train = hypercore_csv.HRCevt1(file_dir+'science_dataframe_'+str(bg_obsid_train)+'.csv')
    bg_test = hypercore_csv.HRCevt1(file_dir+'science_dataframe_'+str(bg_obsid_test)+'.csv')
    
    print(bg_test.data)
    
    
    
    if evt2_filter:
        print('EVT2 filter')
        data_bg_train, data_bg_train_dirty = investigate_masks(bg_train.data.dropna().sample(n=1000), radius_cutout)
        data_bg_test, data_bg_train_dirty = investigate_masks(bg_train.data.dropna().sample(n=1000), radius_cutout)
    else:
        data_bg_train = bg_train.data.dropna()
        data_bg_test = bg_test.data.dropna()
        
    
    
    
    
    
    if run_hyperbola:
        # Test 1: Murray~~~~~

        ## Now plot what happenes with the Murray classification:
        mask_steve = data_clean['Hyperbola test failed'].values
        print('~~~~~~~~~Results from Murray~~~~~~~')
        
        print('mask_steve', type(mask_steve), mask_steve)
        print('~mask_steve', np.invert(mask_steve), ~mask_steve)#, ~list(mask_steve))
        

        df_all = data_clean.dropna()
        df = data_clean[np.invert(mask_steve)].dropna()
        # Okay so these are accepted, zeros
        labels_steve = []
        
        
        df_dropped = data_clean[mask_steve].dropna()
        for k in range(len(df_dropped)):
            labels_steve.append(1)
        
        df_all = pd.concat([ df_dropped, df])
        
        # Def make sure to put the reals on top
        for k in range(len(df)):
            labels_steve.append(0)#0 for x in df]
        
        

        # Then do the cutout selection

        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_all_select = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]
        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]

        print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        sns.set_style('dark')
        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Murray')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        sns.set_style('dark')
        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Rejected Murray')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
        
        
        
        #~~~~~~~~~~~~~~~~~~
        # BUT YOU NEED TO MAKE AN ARRAY THAT IS THE MURRAY CLASSIFICATION RESULT
        xs = df_all['fb_u'].values
        ys = df_all['fp_u'].values
        cs = labels_steve

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['pha'].values
        ys = df_all['sumamps'].values
        cs = labels_steve

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['x'].values
        ys = df_all['y'].values
        cs = labels_steve

        '''plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()'''
        
        
        
        

        #~~~~~~~~~~~~~~~~~~~~~

        ## Now, do the same thing but for the background




        # Test 1: Murray~~~~~

        ## Now plot what happenes with the Murray classification:
        mask_steve = data_bg_test['Hyperbola test failed']
        print('~~~~~~~~~Results from Murray~~~~~~~')


        df_all = data_bg_test
        df = data_bg_test[~mask_steve].dropna()
        df_dropped = data_bg_test[mask_steve].dropna()
        
       
        
        
        
        
        

        # Then do the cutout selection

        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_all_select = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]
        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]

        print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]






        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Accepted Murray')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]






        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Rejected Murray')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()


        # Test 2: Grant~~~~~~
        ## Now run hyperscreen
        print('~~~~~~~~~~~~~~~~~~Hyperscreen~~~~~~~~~~~~~~~~~~~')
        tremblay = obs.hyperscreen()

        mask_grant = tremblay['All Survivals (boolean mask)']


        df = obs.data[mask_grant].dropna()
        df_all = obs.data.dropna()
        df_dropped = obs.data[~mask_grant].dropna()
        
        df_all = pd.concat([df_dropped, df])
        
        labels_grant = []
        for k in range(len(df_dropped)):
            labels_grant.append(1)
        for k in range(len(df)):
            labels_grant.append(0)


        # Then do the cutout selection
        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        df_select_all = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        print('fraction retrieved', len(df_select)/len(df_select_all))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]




        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Grant')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]




        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Rejected Grant')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
        
        #~~~~~~~~~~~~~~~~~~
        # BUT YOU NEED TO MAKE AN ARRAY THAT IS THE MURRAY CLASSIFICATION RESULT
        xs = df_all['fb_u'].values
        ys = df_all['fp_u'].values
        cs = labels_grant

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['pha'].values
        ys = df_all['sumamps'].values
        cs = labels_grant

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['x'].values
        ys = df_all['y'].values
        cs = labels_grant

        '''plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()'''
        

        #~~~~~~~~~~~~~~~~~~~~~


        # Test 2: Grant~~~~~~
        ## Now run hyperscreen
        print('~~~~~~~~~~~~~~~~~~Hyperscreen~~~~~~~~~~~~~~~~~~~')
        tremblay = bg_test.hyperscreen()

        mask_grant = tremblay['All Survivals (boolean mask)']


        df = bg_test.data[mask_grant].dropna()
        df_all = bg_test.data.dropna()
        df_dropped = bg_test.data[~mask_grant].dropna()

        # Then do the cutout selection
        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        df_select_all = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        print('fraction retrieved', len(df_select)/len(df_select_all))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]



        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Accepted Grant')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]



        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Rejected Grant')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()
    
    
    
    # Test 3: Random Forest Baseline~~~~~~~~
    
    result_RFR, result_bag, sci_train, bg_train = prep_data_ML_sci(science_obsid_train, bg_obsid_train, subsamplesize, NP, balanced=balanced)
    
    # This is for testing
    result_test, result_test_bag, sci_test, bg_test = prep_data_ML_sci(science_obsid_test, bg_obsid_test, subsamplesize, NP, balanced=balanced)
    # Cut down sci and bg because they are too big!
    bg = bg_test.sample(n=NP, random_state=1)
    sci = sci_test.sample(n=subsamplesize, random_state=1)
    
    
    
    
    
    random_numbers = np.random.random(size=len(result_RFR))

    result_RFR['random'] = random_numbers
    
    random_numbers = np.random.random(size=len(result_bag))

    result_bag['random'] = random_numbers
    
    random_numbers = np.random.random(size=len(sci_test))

    sci_test['random'] = random_numbers

    random_numbers = np.random.random(size=len(bg_test))

    bg_test['random'] = random_numbers
    
    # Now apply the classification to the entire cutout:
    classify_sci = sci_test.dropna()
    classify_bg = bg_test.dropna()
    
    
    if run_forest:
        print('~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~')
        #['fp_u', 'fb_u', 'fp_v', 'fb_v','pha', 'sumamps', 'random']
        X = result_RFR[feature_list_RFR]



        terms_RFR, reject_terms_RFR, model_RFR = run_RFR(result_RFR, feature_list_RFR, 'no')




        preds_sci_RFR = model_RFR.predict(classify_sci[feature_list_RFR])
        print('sci rfr', preds_sci_RFR)



        # Now make the plot

        # < 0.5 is sci
        ind_select = np.where(np.array(preds_sci_RFR) < th)[0]
        ind_select_drop = np.where(np.array(preds_sci_RFR) > th)[0]

        df = classify_sci.iloc[ind_select]
        df_dropped = classify_sci.iloc[ind_select_drop]
        print('len of selected', len(df), 'len rejected', len(df_dropped))
        print('len before selected', len(classify_sci))


        df_all = classify_sci[np.sqrt((classify_sci['x'] - np.mean(classify_sci['x']))**2 + (classify_sci['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(classify_sci['x']))**2 + (df['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        #print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        print('fraction retrieved', len(df_select)/len(df_all))
        print('len of each of these', len(df_select), len(df_all))

        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Random Forest')
        plt.show()

        x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
        except:
            print('len of xs', len(x))
        plt.title(sci_label+' Rejected Random Forest')
        plt.show()
    
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
    
        # Test 3: Baseline random forest~~~~~~~~
        print('~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~')
        # > 0.5 is background
        preds_bg_RFR = model_RFR.predict(classify_bg[feature_list_RFR])

        ind_bg = np.where(np.array(preds_bg_RFR) < th)[0]
        ind_bg_drop = np.where(np.array(preds_bg_RFR) > th)[0]


        df = classify_bg.iloc[ind_bg]
        df_dropped = classify_bg.iloc[ind_bg_drop]




        df_all = classify_bg[np.sqrt((classify_bg['x'] - np.mean(classify_bg['x']))**2 + (classify_bg['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(classify_bg['x']))**2 + (df['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        #print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        print('fraction retrieved', len(df_select)/len(df_all))
        print('lens', len(df_select), len(df_all))

        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
            plt.title(background_label+' Accepted Random Forest')
            plt.show()
        except:
            print('nothing here', len(y))

        x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
            plt.title(background_label+' Rejected Random Forest')
            plt.show()
        except:
            print('nothing here', len(y))
            
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()
    
    
    # Test 4: PUbag
    print('~~~~~~Bagging PU~~~~~~~')
    feature_list_bag = predictor_list#['fp_u', 'fb_u', 'fp_v', 'fb_v','pha', 'sumamps']
    X = result_bag[feature_list_bag]
    y = result_bag['class'].values
    
    n_estimators, max_samples, len_features = bagging_hyperparameters#bagging_hyperparameters
    
    if max_samples=='balance':# this means you want to do balanced sampling
        max_samples = sum(y>0)
    
    print('length of background examples', sum(y>0), 'length of science', len(y) - sum(y>0), 'total length y',len(y))
    print('# of decision trees', n_estimators)
    print('# of unlabeled samples to draw for each', max_samples)
    print('# of features', len_features)
    
    bc = BaggingPuClassifier(base_estimator=None, 
                             n_estimators=n_estimators,# this is the number of decision trees (SVMs) 
                             max_samples=max_samples, # this is the number of unlabeled samples or K to draw
                             max_features=len_features, 
                             bootstrap=True, 
                             bootstrap_features=False, 
                             oob_score=True, 
                             warm_start=False, 
                             n_jobs=-1, 
                             random_state=True, 
                             verbose=1)


    bc.fit(X, y)
    

    # Store the scores assigned by this approach
    results_bag = pd.DataFrame({
        'label'      : y},        # The labels to be shown to models in experiment
        columns = ['label'])
    results_bag['output_bag'] = bc.oob_decision_function_[:,1]


    # Now use it to predict the test case
    pred_sci_bag = bc.predict_proba(classify_sci[feature_list_bag])[:,1]
    pred_bg_bag = bc.predict_proba(classify_bg[feature_list_bag])[:,1]
    
    if adjust_th:
        #then th will be equal to something else
        
        th = compute_thresh(classify_sci, classify_bg, pred_sci_bag, pred_bg_bag, radius_cutout)
        print('new thresh', th)
        
 
        
    
    
    #~~~~~~~~~~~~~~~~~~~`~~~~~~~~~~~~
    # Now that I have an adjusted threshold, I wonder if its possible to set the midpoint
    xs = classify_sci['fb_u'].values
    ys = classify_sci['fp_u'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,u}$')
    plt.ylabel(r'$f_{p,u}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    xs = classify_sci['fb_v'].values
    ys = classify_sci['fp_v'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    
    xs = classify_sci['pha'].values
    ys = classify_sci['sumamps'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$PHA$')
    plt.ylabel(r'$SUMAMPS$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    xs = classify_sci['x'].values
    ys = classify_sci['y'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    

    indices_sci = np.where(np.array(pred_sci_bag) < th)[0]
    indices_sci_drop = np.where(np.array(pred_sci_bag) > th)[0]
    df = classify_sci.iloc[indices_sci]
    df_dropped = classify_sci.iloc[indices_sci_drop]
    
    
    df_all = classify_sci[np.sqrt((classify_sci['x'] - np.mean(classify_sci['x']))**2 + (classify_sci['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
    df_select = df[np.sqrt((df['x'] - np.mean(classify_sci['x']))**2 + (df['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
    
    #print('same thing but for cutout selection',
    #      'len all', len(df_all_select), 'len selected', len(df_select))
    #print('fraction retrieved', len(df_select)/len(df_all_select))
    
    x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    print('fraction retrieved', len(df_select)/len(df_all))
    print('len', len(df_select), len(df_all))
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data==0, img_data),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Accepted PUBagging')
    plt.show()
    
    x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        plt.colorbar()
    except:
        print(len(x))
    plt.title(sci_label+' Rejected PUBagging')
    plt.show()
    
    plt.clf()
    plt.imshow(img_data/img_data_rej,  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Ratio of accepted to rejected')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Now use it to predict the test case
    pred_bg_bag = bc.predict_proba(classify_bg[feature_list_bag])[:,1]
    
    
    indices_bg_bag = np.where(np.array(pred_bg_bag) < th)
    indices_bg_bag_drop = np.where(np.array(pred_bg_bag) > th)
    df_bag = classify_bg.iloc[indices_bg_bag]
    df_dropped = classify_bg.iloc[indices_bg_bag_drop]
    
    
    #~~~~~~~~~~~~~~~~~~
    
    xs = classify_bg['fb_u'].values
    ys = classify_bg['fp_u'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,u}$')
    plt.ylabel(r'$f_{p,u}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    xs = classify_bg['fb_v'].values
    ys = classify_bg['fp_v'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    xs = classify_bg['pha'].values
    ys = classify_bg['sumamps'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$PHA$')
    plt.ylabel(r'$SUMAMPS$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    '''xs = classify_bg['x'].values
    ys = classify_bg['y'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'magma', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()'''
    
    
    #~~~~~~~~~~~~~~~~~~~~~
    
    df_all_bag = classify_bg[np.sqrt((classify_bg['x'] - np.mean(classify_bg['x']))**2 + (classify_bg['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
    df_select_bag = df_bag[np.sqrt((df_bag['x'] - np.mean(classify_bg['x']))**2 + (df_bag['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
    
    #print('same thing but for cutout selection',
    #      'len all', len(df_all_select), 'len selected', len(df_select))
    #print('fraction retrieved', len(df_select)/len(df_all_select))
    
    x_bag = df_bag['x'][np.sqrt((df_bag['x'] - np.mean(df_bag['x']))**2 + (df_bag['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    y_bag = df_bag['y'][np.sqrt((df_bag['x'] - np.mean(df_bag['x']))**2 + (df_bag['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_bag, yedges_bag, xedges_bag = np.histogram2d(y_bag, x_bag, nbins)
    extent = [xedges_bag[0], xedges_bag[-1], yedges_bag[0], yedges_bag[-1]]

    print('fraction retrieved', len(df_select_bag)/len(df_all_bag))
    print(len(df_select_bag), len(df_all_bag))
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data_bag==0, img_data_bag),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.colorbar()
    plt.title(sci_label+' Accepted PUBagging')
    plt.show()
    
    x_bag = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_bag['x']))**2 + (df_dropped['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    y_bag = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_bag['x']))**2 + (df_dropped['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_bag_rej, yedges_bag, xedges_bag = np.histogram2d(y_bag, x_bag, nbins)
    extent = [xedges_bag[0], xedges_bag[-1], yedges_bag[0], yedges_bag[-1]]

    
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data_bag_rej==0, img_data_bag_rej),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.colorbar()
    plt.title(sci_label+' Rejected PUBagging')
    plt.show()
    
    plt.clf()
    plt.imshow(img_data_bag/img_data_bag_rej,  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Ratio of accepted to rejected')
    plt.show()
    
    if run_forest ==True:
    
        plt.clf()
        plt.hist(preds_sci_RFR, label='Science probabilities', alpha=0.5, bins=30)
        plt.hist(preds_bg_RFR, label='Background probabilities', alpha=0.5, bins=30)
        plt.legend()
        plt.title('RFR')
        plt.show()
    
    plt.clf()
    plt.hist(pred_sci_bag, label='Science probabilities', alpha=0.5, bins=30)
    plt.hist(pred_bg_bag, label='Background probabilities', alpha=0.5, bins=30)
    plt.axvline(x=th)
    plt.legend()
    plt.title('PU Bagging')
    plt.show()
    if run_forest:
        return preds_sci_RFR, preds_bg_RFR, pred_sci_bag, pred_bg_bag, classify_sci, classify_bg
    else:
        return 0, 0, pred_sci_bag, pred_bg_bag, classify_sci, classify_bg








def test_comparison_sci_cutout(science_obsid_train, bg_obsid_train,
                                            science_obsid_test, bg_obsid_test,
                                            sci_label, 
                                            background_label, 
                                            radius_cutout,
                                            nbins,
                                            subsamplesize,
                                            NP,# number of positives for bagging
                                            th,
                                            bagging_hyperparameters,       
                                            predictor_list,# this is for the bag
                                            feature_list_RFR,
                                            run_hyperbola = True,
                                            run_forest = True,
                                            balanced = True,
                                            evt2_filter = False,
                                            adjust_th = True):
    
    
    
    

    file_dir = "/Users/beckynevin/CfA_Code/HRC/hyperscreen/notebooks/csv_files/"
    

    obs = hypercore_csv.HRCevt1(file_dir+'science_dataframe_'+str(science_obsid_test)+'.csv')
    
    
    
    
    # Okay step one is to see if this output has anything that should me excluded going
    # into event2 files
    if evt2_filter:
        data_clean, data_dirty = investigate_masks(obs.data.dropna().sample(n=1000), radius_cutout)#.sample(n=1000)
    else:
        data_clean = obs.data.dropna()
    
    
    bg_train = hypercore_csv.HRCevt1(file_dir+'science_dataframe_cutout_'+str(bg_obsid_train)+'.csv')
    bg_test = hypercore_csv.HRCevt1(file_dir+'science_dataframe_cutout_'+str(bg_obsid_test)+'.csv')
    
    print(bg_test.data)
    
    
    
    if evt2_filter:
        print('EVT2 filter')
        data_bg_train, data_bg_train_dirty = investigate_masks(bg_train.data.dropna().sample(n=1000), radius_cutout)
        data_bg_test, data_bg_train_dirty = investigate_masks(bg_train.data.dropna().sample(n=1000), radius_cutout)
    else:
        data_bg_train = bg_train.data.dropna()
        data_bg_test = bg_test.data.dropna()
        
    
    
    
    
    
    if run_hyperbola:
        # Test 1: Murray~~~~~

        ## Now plot what happenes with the Murray classification:
        mask_steve = data_clean['Hyperbola test failed'].values
        print('~~~~~~~~~Results from Murray~~~~~~~')
        
        print('mask_steve', type(mask_steve), mask_steve)
        print('~mask_steve', np.invert(mask_steve), ~mask_steve)#, ~list(mask_steve))
        

        df_all = data_clean.dropna()
        df = data_clean[np.invert(mask_steve)].dropna()
        # Okay so these are accepted, zeros
        labels_steve = []
        
        
        df_dropped = data_clean[mask_steve].dropna()
        for k in range(len(df_dropped)):
            labels_steve.append(1)
        
        df_all = pd.concat([ df_dropped, df])
        
        # Def make sure to put the reals on top
        for k in range(len(df)):
            labels_steve.append(0)#0 for x in df]
        
        

        # Then do the cutout selection

        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_all_select = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]
        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]

        print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        sns.set_style('dark')
        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Murray')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        sns.set_style('dark')
        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Rejected Murray')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
        
        
        
        #~~~~~~~~~~~~~~~~~~
        # BUT YOU NEED TO MAKE AN ARRAY THAT IS THE MURRAY CLASSIFICATION RESULT
        xs = df_all['fb_u'].values
        ys = df_all['fp_u'].values
        cs = labels_steve

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['pha'].values
        ys = df_all['sumamps'].values
        cs = labels_steve

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['x'].values
        ys = df_all['y'].values
        cs = labels_steve

        '''plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()'''
        
        
        
        

        #~~~~~~~~~~~~~~~~~~~~~

        ## Now, do the same thing but for the background




        # Test 1: Murray~~~~~

        ## Now plot what happenes with the Murray classification:
        mask_steve = data_bg_test['Hyperbola test failed']
        print('~~~~~~~~~Results from Murray~~~~~~~')


        df_all = data_bg_test
        print('trying to mask this', data_bg_test)
        print('with this mask', mask_steve)
        print('this is the inverse mask', ~mask_steve)
        df = data_bg_test[~mask_steve].dropna()
        df_dropped = data_bg_test[mask_steve].dropna()
        
       
        
        
        
        
        

        # Then do the cutout selection

        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_all_select = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]
        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]

        print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]






        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Accepted Murray')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]






        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Rejected Murray')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()


        # Test 2: Grant~~~~~~
        ## Now run hyperscreen
        print('~~~~~~~~~~~~~~~~~~Hyperscreen~~~~~~~~~~~~~~~~~~~')
        tremblay = obs.hyperscreen()

        mask_grant = tremblay['All Survivals (boolean mask)']


        df = obs.data[mask_grant].dropna()
        df_all = obs.data.dropna()
        df_dropped = obs.data[~mask_grant].dropna()
        
        df_all = pd.concat([df_dropped, df])
        
        labels_grant = []
        for k in range(len(df_dropped)):
            labels_grant.append(1)
        for k in range(len(df)):
            labels_grant.append(0)


        # Then do the cutout selection
        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        df_select_all = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        print('fraction retrieved', len(df_select)/len(df_select_all))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]




        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Grant')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]




        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Rejected Grant')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
        
        #~~~~~~~~~~~~~~~~~~
        # BUT YOU NEED TO MAKE AN ARRAY THAT IS THE MURRAY CLASSIFICATION RESULT
        xs = df_all['fb_u'].values
        ys = df_all['fp_u'].values
        cs = labels_grant

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['pha'].values
        ys = df_all['sumamps'].values
        cs = labels_grant

        plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()

        xs = df_all['x'].values
        ys = df_all['y'].values
        cs = labels_grant

        '''plt.clf()

        im2 = plt.scatter(
            xs, ys, 
            c = cs, linewidth = 0, s = 5, alpha = 0.95, 
            vmax=1, vmin=0
        )
        plt.colorbar(im2)
        plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
        plt.show()'''
        

        #~~~~~~~~~~~~~~~~~~~~~


        # Test 2: Grant~~~~~~
        ## Now run hyperscreen
        print('~~~~~~~~~~~~~~~~~~Hyperscreen~~~~~~~~~~~~~~~~~~~')
        tremblay = bg_test.hyperscreen()

        mask_grant = tremblay['All Survivals (boolean mask)']


        df = bg_test.data[mask_grant].dropna()
        df_all = bg_test.data.dropna()
        df_dropped = bg_test.data[~mask_grant].dropna()

        # Then do the cutout selection
        mean_x = np.mean(df_all.x)
        mean_y = np.mean(df_all.y)

        df_select = df[np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        df_select_all = df_all[np.sqrt((df_all.x - mean_x)**2 + (df_all.y - mean_y)**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        print('fraction retrieved', len(df_select)/len(df_select_all))

        x = df['x'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]
        y = df['y'][np.sqrt((df.x - mean_x)**2 + (df.y - mean_y)**2) < radius_cutout]



        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Accepted Grant')
        plt.show()
        
        x = df_dropped['x'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped.x - mean_x)**2 + (df_dropped.y - mean_y)**2) < radius_cutout]



        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Rejected Grant')
        plt.show()
        
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()
    
    
    
    # Test 3: Random Forest Baseline~~~~~~~~
    
    result_RFR, result_bag, sci_train, bg_train = prep_data_ML_sci(science_obsid_train, bg_obsid_train, subsamplesize, NP, balanced=balanced)
    
    # This is for testing
    result_test, result_test_bag, sci_test, bg_test = prep_data_ML_sci(science_obsid_test, bg_obsid_test, subsamplesize, NP, balanced=balanced)
    # Cut down sci and bg because they are too big!
    bg = bg_test.sample(n=NP, random_state=1)
    sci = sci_test.sample(n=subsamplesize, random_state=1)
    
    
    
    
    
    random_numbers = np.random.random(size=len(result_RFR))

    result_RFR['random'] = random_numbers
    
    random_numbers = np.random.random(size=len(result_bag))

    result_bag['random'] = random_numbers
    
    random_numbers = np.random.random(size=len(sci_test))

    sci_test['random'] = random_numbers

    random_numbers = np.random.random(size=len(bg_test))

    bg_test['random'] = random_numbers
    
    # Now apply the classification to the entire cutout:
    classify_sci = sci_test.dropna()
    classify_bg = bg_test.dropna()
    
    
    if run_forest:
        print('~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~')
        #['fp_u', 'fb_u', 'fp_v', 'fb_v','pha', 'sumamps', 'random']
        X = result_RFR[feature_list_RFR]



        terms_RFR, reject_terms_RFR, model_RFR = run_RFR(result_RFR, feature_list_RFR, 'no')




        preds_sci_RFR = model_RFR.predict(classify_sci[feature_list_RFR])
        print('sci rfr', preds_sci_RFR)



        # Now make the plot

        # < 0.5 is sci
        ind_select = np.where(np.array(preds_sci_RFR) < th)[0]
        ind_select_drop = np.where(np.array(preds_sci_RFR) > th)[0]

        df = classify_sci.iloc[ind_select]
        df_dropped = classify_sci.iloc[ind_select_drop]
        print('len of selected', len(df), 'len rejected', len(df_dropped))
        print('len before selected', len(classify_sci))


        df_all = classify_sci[np.sqrt((classify_sci['x'] - np.mean(classify_sci['x']))**2 + (classify_sci['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(classify_sci['x']))**2 + (df['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        #print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        print('fraction retrieved', len(df_select)/len(df_all))
        print('len of each of these', len(df_select), len(df_all))

        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Accepted Random Forest')
        plt.show()

        x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
        except:
            print('len of xs', len(x))
        plt.title(sci_label+' Rejected Random Forest')
        plt.show()
    
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(sci_label+' Ratio of accepted to rejected')
        plt.show()
    
        # Test 3: Baseline random forest~~~~~~~~
        print('~~~~~~~~~~~~~~~~~ Random Forest ~~~~~~~~~~~~~~~')
        # > 0.5 is background
        preds_bg_RFR = model_RFR.predict(classify_bg[feature_list_RFR])

        ind_bg = np.where(np.array(preds_bg_RFR) < th)[0]
        ind_bg_drop = np.where(np.array(preds_bg_RFR) > th)[0]


        df = classify_bg.iloc[ind_bg]
        df_dropped = classify_bg.iloc[ind_bg_drop]




        df_all = classify_bg[np.sqrt((classify_bg['x'] - np.mean(classify_bg['x']))**2 + (classify_bg['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
        df_select = df[np.sqrt((df['x'] - np.mean(classify_bg['x']))**2 + (df['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]

        #print('same thing but for cutout selection',
        #      'len all', len(df_all_select), 'len selected', len(df_select))
        #print('fraction retrieved', len(df_select)/len(df_all_select))

        x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        print('fraction retrieved', len(df_select)/len(df_all))
        print('lens', len(df_select), len(df_all))

        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
            plt.title(background_label+' Accepted Random Forest')
            plt.show()
        except:
            print('nothing here', len(y))

        x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
        y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]





        img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        try:
            plt.colorbar()
            plt.title(background_label+' Rejected Random Forest')
            plt.show()
        except:
            print('nothing here', len(y))
            
        plt.clf()
        plt.imshow(img_data/img_data_rej,  
                        rasterized=True, cmap='viridis', origin='data', extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        plt.xlabel('x')
        plt.ylabel('y')

        plt.colorbar()
        plt.title(background_label+' Ratio of accepted to rejected')
        plt.show()
    
    
    # Test 4: PUbag
    print('~~~~~~Bagging PU~~~~~~~')
    feature_list_bag = predictor_list#['fp_u', 'fb_u', 'fp_v', 'fb_v','pha', 'sumamps']
    X = result_bag[feature_list_bag]
    y = result_bag['class'].values
    
    n_estimators, max_samples, len_features = bagging_hyperparameters#bagging_hyperparameters
    
    if max_samples=='balance':# this means you want to do balanced sampling
        max_samples = sum(y>0)
    
    print('length of background examples', sum(y>0), 'length of science', len(y) - sum(y>0), 'total length y',len(y))
    print('# of decision trees', n_estimators)
    print('# of unlabeled samples to draw for each', max_samples)
    print('# of features', len_features)
    
    bc = BaggingPuClassifier(base_estimator=None, 
                             n_estimators=n_estimators,# this is the number of decision trees (SVMs) 
                             max_samples=max_samples, # this is the number of unlabeled samples or K to draw
                             max_features=len_features, 
                             bootstrap=True, 
                             bootstrap_features=False, 
                             oob_score=True, 
                             warm_start=False, 
                             n_jobs=-1, 
                             random_state=True, 
                             verbose=1)


    bc.fit(X, y)
    

    # Store the scores assigned by this approach
    results_bag = pd.DataFrame({
        'label'      : y},        # The labels to be shown to models in experiment
        columns = ['label'])
    results_bag['output_bag'] = bc.oob_decision_function_[:,1]


    # Now use it to predict the test case
    pred_sci_bag = bc.predict_proba(classify_sci[feature_list_bag])[:,1]
    pred_bg_bag = bc.predict_proba(classify_bg[feature_list_bag])[:,1]
    
    if adjust_th:
        #then th will be equal to something else
        
        th = compute_thresh(classify_sci, classify_bg, pred_sci_bag, pred_bg_bag, radius_cutout)
        print('new thresh', th)
        
 
        
    
    
    #~~~~~~~~~~~~~~~~~~~`~~~~~~~~~~~~
    # Now that I have an adjusted threshold, I wonder if its possible to set the midpoint
    xs = classify_sci['fb_u'].values
    ys = classify_sci['fp_u'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,u}$')
    plt.ylabel(r'$f_{p,u}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    xs = classify_sci['fb_v'].values
    ys = classify_sci['fp_v'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    
    xs = classify_sci['pha'].values
    ys = classify_sci['sumamps'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$PHA$')
    plt.ylabel(r'$SUMAMPS$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    xs = classify_sci['x'].values
    ys = classify_sci['y'].values
    cs = pred_sci_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on sci')
    plt.show()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    

    indices_sci = np.where(np.array(pred_sci_bag) < th)[0]
    indices_sci_drop = np.where(np.array(pred_sci_bag) > th)[0]
    df = classify_sci.iloc[indices_sci]
    df_dropped = classify_sci.iloc[indices_sci_drop]
    
    
    df_all = classify_sci[np.sqrt((classify_sci['x'] - np.mean(classify_sci['x']))**2 + (classify_sci['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
    df_select = df[np.sqrt((df['x'] - np.mean(classify_sci['x']))**2 + (df['y'] - np.mean(classify_sci['y']))**2) < radius_cutout]
    
    #print('same thing but for cutout selection',
    #      'len all', len(df_all_select), 'len selected', len(df_select))
    #print('fraction retrieved', len(df_select)/len(df_all_select))
    
    x = df['x'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    y = df['y'][np.sqrt((df['x'] - np.mean(df_all['x']))**2 + (df['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    print('fraction retrieved', len(df_select)/len(df_all))
    print('len', len(df_select), len(df_all))
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data==0, img_data),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Accepted PUBagging')
    plt.show()
    
    x = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    y = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_all['x']))**2 + (df_dropped['y'] - np.mean(df_all['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_rej, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(ma.masked_where(img_data_rej==0, img_data_rej),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        plt.colorbar()
    except:
        print(len(x))
    plt.title(sci_label+' Rejected PUBagging')
    plt.show()
    
    plt.clf()
    plt.imshow(img_data/img_data_rej,  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Ratio of accepted to rejected')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Now use it to predict the test case
    pred_bg_bag = bc.predict_proba(classify_bg[feature_list_bag])[:,1]
    
    
    indices_bg_bag = np.where(np.array(pred_bg_bag) < th)
    indices_bg_bag_drop = np.where(np.array(pred_bg_bag) > th)
    df_bag = classify_bg.iloc[indices_bg_bag]
    df_dropped = classify_bg.iloc[indices_bg_bag_drop]
    
    
    #~~~~~~~~~~~~~~~~~~
    
    xs = classify_bg['fb_u'].values
    ys = classify_bg['fp_u'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,u}$')
    plt.ylabel(r'$f_{p,u}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    xs = classify_bg['fb_v'].values
    ys = classify_bg['fp_v'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    xs = classify_bg['pha'].values
    ys = classify_bg['sumamps'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'RdBu', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$PHA$')
    plt.ylabel(r'$SUMAMPS$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()
    
    '''xs = classify_bg['x'].values
    ys = classify_bg['y'].values
    cs = pred_bg_bag

    plt.clf()

    im2 = plt.scatter(
        xs, ys, 
        c = cs, linewidth = 0, s = 5, alpha = 0.95, 
        cmap = 'magma', norm = matplotlib.colors.DivergingNorm(vmin=0, vcenter=th, vmax=1)
    )
    plt.colorbar(im2)
    plt.xlabel(r'$f_{b,v}$')
    plt.ylabel(r'$f_{p,v}$')
    plt.title(r'Using ${\tt BaggingClassifierPU}$ on stowed')
    plt.show()'''
    
    
    #~~~~~~~~~~~~~~~~~~~~~
    
    df_all_bag = classify_bg[np.sqrt((classify_bg['x'] - np.mean(classify_bg['x']))**2 + (classify_bg['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
    df_select_bag = df_bag[np.sqrt((df_bag['x'] - np.mean(classify_bg['x']))**2 + (df_bag['y'] - np.mean(classify_bg['y']))**2) < radius_cutout]
    
    #print('same thing but for cutout selection',
    #      'len all', len(df_all_select), 'len selected', len(df_select))
    #print('fraction retrieved', len(df_select)/len(df_all_select))
    
    x_bag = df_bag['x'][np.sqrt((df_bag['x'] - np.mean(df_bag['x']))**2 + (df_bag['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    y_bag = df_bag['y'][np.sqrt((df_bag['x'] - np.mean(df_bag['x']))**2 + (df_bag['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_bag, yedges_bag, xedges_bag = np.histogram2d(y_bag, x_bag, nbins)
    extent = [xedges_bag[0], xedges_bag[-1], yedges_bag[0], yedges_bag[-1]]

    print('fraction retrieved', len(df_select_bag)/len(df_all_bag))
    print(len(df_select_bag), len(df_all_bag))
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data_bag==0, img_data_bag),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.colorbar()
    plt.title(sci_label+' Accepted PUBagging')
    plt.show()
    
    x_bag = df_dropped['x'][np.sqrt((df_dropped['x'] - np.mean(df_bag['x']))**2 + (df_dropped['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    y_bag = df_dropped['y'][np.sqrt((df_dropped['x'] - np.mean(df_bag['x']))**2 + (df_dropped['y'] - np.mean(df_bag['y']))**2) < radius_cutout]
    
    
    
    
    
    img_data_bag_rej, yedges_bag, xedges_bag = np.histogram2d(y_bag, x_bag, nbins)
    extent = [xedges_bag[0], xedges_bag[-1], yedges_bag[0], yedges_bag[-1]]

    
    
    plt.clf()
    plt.imshow(ma.masked_where(img_data_bag_rej==0, img_data_bag_rej),  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.colorbar()
    plt.title(sci_label+' Rejected PUBagging')
    plt.show()
    
    plt.clf()
    plt.imshow(img_data_bag/img_data_bag_rej,  
                    rasterized=True, cmap='viridis', origin='data', extent=extent, 
                   norm=matplotlib.colors.LogNorm())
    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar()
    plt.title(sci_label+' Ratio of accepted to rejected')
    plt.show()
    
    if run_forest ==True:
    
        plt.clf()
        plt.hist(preds_sci_RFR, label='Science probabilities', alpha=0.5, bins=30)
        plt.hist(preds_bg_RFR, label='Background probabilities', alpha=0.5, bins=30)
        plt.legend()
        plt.title('RFR')
        plt.show()
    
    plt.clf()
    plt.hist(pred_sci_bag, label='Science probabilities', alpha=0.5, bins=30)
    plt.hist(pred_bg_bag, label='Background probabilities', alpha=0.5, bins=30)
    plt.axvline(x=th)
    plt.legend()
    plt.title('PU Bagging')
    plt.show()
    if run_forest:
        return preds_sci_RFR, preds_bg_RFR, pred_sci_bag, pred_bg_bag, classify_sci, classify_bg
    else:
        return 0, 0, pred_sci_bag, pred_bg_bag, classify_sci, classify_bg
        
def prep_data_ML_sci(obsid, obsid_bg, n_points, NP, balanced=True):
    file_dir = "/Users/beckynevin/CfA_Code/HRC/hyperscreen/notebooks/csv_files/"
    sci = pd.read_csv(file_dir+'science_dataframe_'+str(obsid)+'.csv', index_col=0)
    bg = pd.read_csv(file_dir+'science_dataframe_'+str(obsid_bg)+'.csv', index_col=0)
    
    # First, get rid of the columns that have only NaNs:
    # Right of. thebat make sure you don't need to drop stuff                                                                                                            
    try:
        bg = bg.drop(columns=['time','samp'])
    except:
        try:
            bg = bg.drop(columns='time')
        except:
            print('no columns')

    try:
        sci = sci.drop(columns=['time','samp'])
    except:
        try:
            sci = sci.drop(columns='time')
        except:
            print('no columns')

    
    a_u = bg["au1"]  # otherwise known as "a1"
    b_u = bg["au2"]  # "a2"
    c_u = bg["au3"]  # "a3"

    a_v = bg["av1"]
    b_v = bg["av2"]
    c_v = bg["av3"]

    with np.errstate(invalid='ignore'):
        # Do the U axis
        fp_u = ((c_u - a_u) / (a_u + b_u + c_u))
        fb_u = b_u / (a_u + b_u + c_u)

        # Do the V axis
        fp_v = ((c_v - a_v) / (a_v + b_v + c_v))
        fb_v = b_v / (a_v + b_v + c_v)

    bg['fp_u'] = fp_u
    bg['fb_u'] = fb_u
    bg['fp_v'] = fp_v
    bg['fb_v'] = fb_v
    
    a_u = sci["au1"]  # otherwise known as "a1"
    b_u = sci["au2"]  # "a2"
    c_u = sci["au3"]  # "a3"

    a_v = sci["av1"]
    b_v = sci["av2"]
    c_v = sci["av3"]

    with np.errstate(invalid='ignore'):
        # Do the U axis
        fp_u = ((c_u - a_u) / (a_u + b_u + c_u))
        fb_u = b_u / (a_u + b_u + c_u)

        # Do the V axis
        fp_v = ((c_v - a_v) / (a_v + b_v + c_v))
        fb_v = b_v / (a_v + b_v + c_v)

    sci['fp_u'] = fp_u
    sci['fb_u'] = fb_u
    sci['fp_v'] = fp_v
    sci['fb_v'] = fb_v
    
    
    
     
    
    # Now create class labels, label all of the sci events as 'unlabelled' or '0'
    # and all of the background events as 'positive' or '1'
    bg['class'] = 1
    sci['class'] = 0
    
    
    if balanced==False: #this means we are putting together the training set for the RFR where we want the sample sizes to be equal                                      
        frames_bag = [bg.sample(n=NP, random_state=1), sci.sample(n=n_points, random_state=1)]
        frames_RFR = [bg.sample(n=int(n_points/5), random_state=1), sci.sample(n=n_points, random_state=1)]

        result_bag = pd.concat(frames_bag)
        #drop nans:                                                                                                                                                      
        #result_bag = result_bag.drop(columns='time')                                                                                                                    
        result_bag = result_bag.dropna()

        result_RFR = pd.concat(frames_RFR)
        #drop nans:                                                                                                                                                      
        #result_RFR = result_RFR.drop(columns='time')                                                                                                                    
        result_RFR = result_RFR.dropna()

        return result_RFR, result_bag, sci, bg # result_bag is NP of background and subsamplesize of sci                                                                 
    else:
        frames = [bg.sample(n=n_points, random_state=1), sci.sample(n=n_points, random_state=1)]
        #frames = [bg, sci]                                                                                                                                              
        result = pd.concat(frames)
        #drop nans:                                                                                                                                                      
        #result = result.drop(columns='time')                                                                                                                            
        result = result.dropna()
        return result, result, sci, bg   



        
print('compiled')

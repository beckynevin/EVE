### EVE
<img src="EVE_logo.png">

## Organization of this repo

This repo has multiple

## Installing the ciao_knodle environment

Before running any of the notebooks or downloading event files, you must have ciao-4.13 installed as an anaconda environment. The EVE package also requires hyperscreen, caldb, marx as well as certain pip-installed modules from knodle. I therefore created one combined environment named ciao_knodle. It can be installed using the following line with the environment.yml file in this repo (environment_ciao_plus_knodle.yml):

```
conda env create -f environment_ciao_plus_knodle.yml
```
^This will create a new environment where the name is given by the first line of the environment.yml file. 

Full separate directions for installing ciao and caldb can be found here: https://cxc.cfa.harvard.edu/ciao/download/index.html

And specific instructions for installing ciao with conda can be found here: https://cxc.cfa.harvard.edu/ciao/threads/ciao_install_conda/

To activate your new environment:
```
source activate ciao_knodle
```

It is highly encouraged to run the smoke tests to make sure everything is installed correctly.

If you plan to run any of the notebooks in the notebook folder also add:
```
conda install jupyter notebook
```


## Download event files
To obtain the event files use this ciao command, which drops everything into a folder named by obsid in the current directory:
```
download_chandra_obsid 1505
```
where 1505 is the obsid (in this case for Cas A).

More options can be found here: https://cxc.cfa.harvard.edu/ciao/ahelp/download_chandra_obsid.html

In this case, the event 1 files are the priority (event 2 have flags applied, including the hyperbola selection). To get just event 1 files:
```
download_chandra_obsid XXXX evt1
```
For more information, you can look up observations here: https://cda.harvard.edu/chaser/

## Download stowed background
When you install ciao, the stowed background files are already downloaded! They can be found at $CALDB/data/chandra/hrc/bkgrnd/. <-- For some reason this doesn't register for me.

More details can be found here about how to determine the stowed background file that corresponds to a given observation: https://cxc.harvard.edu/ciao/threads/hrci_bg_events/

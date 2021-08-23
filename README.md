### EVE
<img src="EVE_logo.png">

## Installing CIAO, the Chandra Interactive Analysis of Operations software package, which is necessarly for downloading event files AND for running the notebooks

Before running any of the notebooks, you must have ciao-4.13 installed as an anaconda environment.

Full directions for installing ciao and caldb can be found here: https://cxc.cfa.harvard.edu/ciao/download/index.html

And specific instructions for installing ciao with conda can be found here: https://cxc.cfa.harvard.edu/ciao/threads/ciao_install_conda/

More specifically, this line should do it:
```
conda create -n ciao-4.13 \
  -c https://cxc.cfa.harvard.edu/conda/ciao \
  ciao sherpa ds9 ciao-contrib caldb marx
```
where caldb is important because it enables you to use background files (as opposed to caldb_main)

And then load up the environment:
```
source activate ciao-4.13
```

It is highly encouraged to run the smoke tests to make sure everything is installed correctly.

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

## Download stowed background

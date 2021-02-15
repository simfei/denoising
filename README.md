### script v1 (cmd)

#### 1. install Anaconda
https://docs.anaconda.com/anaconda/install/windows/  
https://docs.anaconda.com/anaconda/install/mac-os/

#### 2. create an environment with python 3.7, activate the environment
``` 
conda create -n [ENV_NAME] python==3.7
conda activate [ENV_NAME]
```

#### 3. go to the directory with requirements.txt, run the following command to install all dependencies 
```
pip install -r requirements.txt
```

#### 4. go to the directory with script.py, run the following command
```
(conda activate [ENV_NAME])
python script.py --imgType=vas --modelName=n2v --imgPath=test_images/VAS/raw/ --savePath=denoised/
```



### script v2 (jupyter notebook)

#### 1. install Anaconda
https://docs.anaconda.com/anaconda/install/windows/  
https://docs.anaconda.com/anaconda/install/mac-os/

#### 2. create an environment with python 3.7, activate the environment, and add the environment to jupyter
``` 
conda create -n [ENV_NAME] python==3.7 
conda activate [ENV_NAME]
conda install -n [ENV_NAME] ipykernel
python -m ipykernel install --user --name [ENV_NAME] --display-name [DISPLAY_NAME]
```
#### * to remove environment from jupyter
```
jupyter kernelspec remove [ENV_NAME]
```

#### 3. go to the directory with requirements.txt, run the following command to install all dependencies 
```
pip install -r requirements.txt
```

#### 4. open script.ipynb in jupyter, change config, run all the cells




### script v3 (colab)

#### 1. install all the dependencies
```
! pip install n2v
! pip uninstall tensorflow
! pip install tensorflow==1.15.0
! pip uninstall gast
! pip install gast
```

#### 2. upload data or mounted at Google Drive

#### 3. change config and run all the cells

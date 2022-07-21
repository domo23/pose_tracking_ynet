## Start
```shell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Note 
For windows installation there might be problem to install OpenExr. One of solution is to manually download it from https://www.lfd.uci.edu/~gohlke/pythonlibs/ and install through pip.

## Training
```shell
usage: train_network.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--object_name OBJECT_NAME] [--ycb_ds_path YCB_DS_PATH] [--validation_scenes VALIDATION_SCENES [VALIDATION_SCENES ...]]

Train YNet network on YCB-Video dataset.

optional arguments:
  -h, --help                    show this help message and exit
  --batch_size BATCH_SIZE
                                batch size for training network, default 8
  --epochs EPOCHS               amount of epochs for training network, default 15
  --object_name OBJECT_NAME
                                name of object for training, default 035_power_drill
  --ycb_ds_path YCB_DS_PATH
                                absolute path to YCB-Video dataset
  --validation_scenes VALIDATION_SCENES [VALIDATION_SCENES ...]
                                validation scenes for training, default ['0050']
```

## Evaluating
```shell
usage: evaluate_network.py [-h] [--object_name OBJECT_NAME] [--ycb_ds_path YCB_DS_PATH] [--validation_scenes VALIDATION_SCENES [VALIDATION_SCENES ...]] [--weights_path WEIGHTS_PATH]

Evaluate YNet network on YCB-Video dataset.

optional arguments:
  -h, --help                    show this help message and exit
  --object_name OBJECT_NAME
                                name of object for training, default 035_power_drill
  --ycb_ds_path YCB_DS_PATH
                                absolute path to YCB-Video dataset
  --validation_scenes VALIDATION_SCENES [VALIDATION_SCENES ...]
                                validation scenes, default ['0050']
  --weights_path WEIGHTS_PATH
                                path with trained weights, default best_weights.hdf5
```
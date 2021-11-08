# Object Detection Setup and Training

## Setup
Create a python virtualenv for training the model  
```
virtualenv -p python3 venv_tf1153
```
Activate the env
```
source venv_tf1153/bin/activate
```
Install the packages from requirements.txt
```
pip install -r requirements.txt
```

## Build packages

### Build proto packages
Install protobuf (if it is not installed)
```
pip install protobuf
```
Build the packages
```
protoc object_detection/protos/*.proto --python_out=.
```

### Build the slim models
Go inside the folder `research/slim`  
And build and install the networks
```
python setup.py build
python setup.py install
```

## Set the Python path
```
export PYTHONPATH=<path-to-slim>:<path-to-research-dir>
```

## Run training
Refer the train.sh script

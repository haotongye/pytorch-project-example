# pytorch-project-example
This repository hosts my usual pipeline and template code for working with a
[PyTorch](https://pytorch.org/) project. An example usage of applying
[transformers](https://github.com/huggingface/transformers) `albert-base-v2` model on
[SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) is provided.


## Requirements
```
ipdb
python-box
tqdm
numpy
pytorch>=1.4
transformers>=2.3
matplotlib
```


## Project Structure

### *common/*
- Model and trainer base classes which supports operations like saving/loading
model checkpoints, continue training, etc. 
- Customizable metrics and losses that work with the trainer.
- Utilities functions like saving/loading json files, setting random seed, etc.

### *model/*
This directory contains task-specific code and config for dataset/model.

### *scripts/*
Useful scripts for data cleaning, evaluation, etc.


## Usage

### Preparation
1. Create directories.
```
mkdir data datasets models
```
2. Download SQuAD 2.0 data
```
wget -P ./data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P ./data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

### Create Dataset
1. Create a dataset directory.
```
mkdir ./datasets/squad2_albert-base-v2/
```
2. Copy template config to the dataset directory.
```
cp ./model/dataset_config_template.yaml ./datasets/squad2_albert-base-v2/config.yaml
```
3. Edit `data_dir` in config. (If you followed the [Preparation](###Preparation) steps
then it should be `./data/`.)
4. Create dataset.
```
python -m model.create_dataset ./datasets/squad2_albert-base-v2/
```
Tokenized datasets *train.pkl* and *dev.pkl* will be saved under
*./datasets/squad2_albert-base-v2/*.

### Train
1. Create a model directory.
```
mkdir ./models/squad2_albert-base-v2/
```
2. Copy template config to the model directory.
```
cp ./model/model_config_template.yaml ./models/squad2_albert-base-v2/config.yaml
```
3. Edit `dataset_dir` in config. (If you followed the
[Create Dataset](###Create-Dataset) steps then it should be
`./datasets/squad2_albert-base-v2/`.)
4. Train the model.
```
python -m model.train ./models/squad2_albert-base-v2/
```
5. Monitor training log.
```
tail -f ./models/squad2_albert-base-v2/log.csv
```
After training is completed, model checkpoints can be found at
*./models/squad2_albert-base-v2/ckpts/*.

### Predict
```
python -m model.predict ./datasets/squad2_albert-base-v2/dev.pkl ./models/squad2_albert-base-v2/ckpts/epoch-3.ckpt
```
Predictions will be saved to `./models/squad2_albert-base-v2/predictions/`.

### Evaluate
Run
```
python scripts/squad2_evaluate.py ./data/dev-v2.0.json ./models/squad2_albert-base-v2/predictions/epoch-3_answer.json --na-prob-file models/squad2_albert-base-v2/predictions/epoch-3_na_prob.json
```
Output:
```
{
    "exact": 76.27389876189675,
    "f1": 80.09090869478733,
    "total": 11873,
    "HasAns_exact": 73.65047233468286,
    "HasAns_f1": 81.29543841653317,
    "HasAns_total": 5928,
    "NoAns_exact": 78.88982338099242,
    "NoAns_f1": 78.88982338099242,
    "NoAns_total": 5945,
    "best_exact": 77.57938179061736,
    "best_exact_thresh": 0.11266700178384781,
    "best_f1": 80.99138345067499,
    "best_f1_thresh": 0.11277145147323608
}
```
(Numbers may differ slightly on your machine.)


## Limitations
Framework in this repo works really well for supervised NLP tasks. It may need some
extensions in order to incorporate more complicated projects like RL or GAN.


## Licence
This project is licensed under the terms of the [MIT license](LICENSE.txt).

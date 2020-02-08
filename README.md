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
python -m model.train ./models/suqad2_albert-base-v2/
```

### Predict
```
python -m model.predict ./datasets/squad2_albert-base-v2/dev.pkl ./models/suqad2_albert-base-v2/ckpts/epoch-3.ckpt
```
Predictions will be saved to `./models/suqad2_albert-base-v2/predictions/`.

### Evaluate
```
python scripts/squad2_evaluate.py ./data/dev-v2.0.json ./models/suqad2_albert-base-v2/predictions/epoch-3_answer.json --na-prob-file models/suqad2_albert-base-v2/predictions/epoch-3_na_prob.json
```


<!-- ## Preprocessed Datasets and Model Checkpoints -->
<!-- You can download prerpcessd datasets and model checkpoints and skip the dataset creation -->
<!-- and training steps. -->


<!-- ## Limitations -->

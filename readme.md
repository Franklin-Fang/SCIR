# SCIR

This repo contains the code for **SCIR: A Self-Correcting Iterative Refinement Framework for Enhanced Information Extraction Based on Schema.** In this paper, we create a multitask bilingual self-review training dataset (MBSF) for self-review model training, and propose a low-training-cost, high-generalization Self-Correcting Iterative Refinement (SCIR) framework.

<a target="_blank" href="https://anonymous.4open.science/r/SCIR">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-black?style=flat&logo=github"></a>
<a target="_blank" href="https://anonymous.4open.science/r/SCIR/data/train/README.md">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://anonymous.4open.science/r/SCIR/model/README.md">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>

<br>

## Architecture  
<img width="716" alt="abs" src="https://anonymous.4open.science/r/SCIR/SCIR.png">

## Getting Started

1. First, clone the repository and download the dataset:

```bash
git clone https://anonymous.4open.science/r/SCIR.git
```

2. Install additional requirements:

```bash
conda create -n scir python=3.12 
conda activate scir
cd ./SCIR
pip install -r requirements.txt
```
## Training Steps

1. First install ms-swift.
2. Configure model in ./model and dataset in ./data

3. Start training:
```bash
cd ./train
bash train_redundancy.sh   #train model to check redundancy  
bash train_missing.sh      #train model to check missing  
```

## Generating Steps

1. Download dataset and make sure the data in the path: ./data/test, or you can use the test data in already in this path.

2. Start generate

```python
cp ./script/run.sh ./
bash ./run.sh  
```

3. Start  evaluation

```python
cp ./script/eval.sh ./
bash ./eval.sh  
```

## Ablation  Experiment

Use the same way as Generating Steps, but you need to change the start script:

```python
#For only use RedundancyCheckModel
cp ./script/ablation/Redundancy.sh ./
bash ./Redundancy.sh   

#For only use MissingCheckModel
cp ./script/ablation/Missing.sh ./
bash ./Missing.sh     
```


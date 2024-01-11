# Models

## Introduction
This directory contains the deep learning recommendation models used in RMixer. The naming rule for each folder is `dataset_model`. 

The models are from [PaddlePaddle/PaddleRec: Large-scale recommendation algorithms library](https://github.com/PaddlePaddle/PaddleRec), and we have done a little change to fit RMixer.

## Directory structure 

The models directory contains several model folders and some runtime tools. 

### Model

The directory structure under each model folder is shown below.

```
├── data #Sample data
  ├── sample_data
├── __init__.py
├── config.yaml # Sample data configuration
├── config_bigdata.yaml # Full data configuration
├── net.py # Model core networking
├── reader.py # Data reader
├── dygraph_model.py # Build dynamic graph
```
### Model List

In RMixer, the models we are involved in mainly consist of the following models.

|   模型    |                             论文                             |
| :-------: | :----------------------------------------------------------: |
|   flen    | [FLEN: Leveraging Field for Scalable CTR Prediction]([[1911.04690\] FLEN: Leveraging Field for Scalable CTR Prediction (arxiv.org)](https://arxiv.org/abs/1911.04690)) |
| bert4rec  | [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://dl.acm.org/doi/abs/10.1145/3357384.3357895) |
|   DCN2    | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems ](https://dl.acm.org/doi/10.1145/3442381.3450078) |
|   difm    | [A dual input-aware factorization machine for CTR prediction](https://dl.acm.org/doi/10.5555/3491440.3491874) |
|   dpin    | [Deep Position-wise Interaction Network for CTR Prediction](https://dl.acm.org/doi/10.1145/3404835.3463117) |
| wide&deep | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)(2016) |
|  DeepFM   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)(2017) |
|    bst    | [Behavior sequence transformer for e-commerce recommendation in alibaba](https://arxiv.org/pdf/1905.06874v1.pdf) |
|    DCN    | [DeepAndCross: Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754) |
|   dien    | [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672v5.pdf) |
|    din    | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978v4.pdf) |
|   dlrm    | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091v1.pdf) |
|    dmr    | [Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://github.com/lvze92/DMR/blob/master/%5BDMR%5D%20Deep%20Match%20to%20Rank%20Model%20for%20Personalized%20Click-Through%20Rate%20Prediction-AAAI20.pdf) |
### Others

In addition to the aforementioned models, this directory also includes the following files:

```
├── __init__.py
├── trainer.py          # An independent model trainer for monolithic model training
├── infer.py            # An independent model inferencer for monolithic model inference
├── rmixer_train.py     # RMixer's model trainer, providing training interface for RMixer
├── mps_train.py        # MPS's model trainer, achieving co-location training using MPS
├── README.md 
```

## Quick start

```bash
# Enter model directory
cd models/xxx # xxx is the model directory under any models
# Training
python -u ../trainer.py -m config.yaml # Run config_bigdata.yaml for full data
# Inference
python -u ../infer.py -m config.yaml 
```


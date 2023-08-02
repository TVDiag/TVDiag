# TVDiag

### TVDiag: A Task-oriented and View-invariant Failure Diagnosis Framework with Multimodal Data

TVDiag is a multimodal failure diagnosis framework designed to locate the root cause and identify the failure type in microservice-based systems. This repository offers the core implementation of TVDiag.


## Project Structure
```
.
├── config
│   └── experiment.yaml
├── core
│   ├── aug.py
│   ├── ita.py
│   ├── loss
│   │   ├── AutomaticWeightedLoss.py
│   │   ├── SupervisedContrastiveLoss.py
│   │   └── UnsupervisedContrastiveLoss.py
│   ├── model
│   │   ├── backbone
│   │   │   ├── FC.py
│   │   │   ├── GAT.py
│   │   │   └── TAGClassify.py
│   │   ├── Classifier.py
│   │   ├── Encoder.py
│   │   └── MainModel.py
│   ├── multimodal_dataset.py
│   └── TVDiag.py
├── data
│   ├── aiops22-pod
│   │   ├── fasttext
│   │   └── raw
│   └── gaia
│       ├── fasttext
│       └── raw
├── helper
│   ├── eval.py
│   ├── io.py
│   ├── paths.py
│   └── timer.py
├── process
│   ├── AIOps22Process.py
│   ├── BaseProcess.py
│   ├── events
│   │   ├── fasttext_with_DA.py
│   │   └── sententce_embedding.py
│   └── GaiaProcess.py
├── requirements.txt
├── README.md
└── main.py

```

## Dataset
We conducted experiments on two datasets:
- [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA dataset records metrics, traces, and logs of the MicroSS simulation system in July 2021, which consists of ten microservices and some middleware such as Redis, MySQL, and Zookeeper. The extracted events of GAIA can be accessible on [DiagFusion](https://arxiv.org/abs/2302.10512).
- [AIOps-22](https://competition.aiops-challenge.com). The AIOps-22 dataset is derived from the training data released by the AIOps 2022 Challenge, where failures at three levels (node, service, and instance) were injected into a Web-based e-commerce platform [Online-boutique](https://github.com/GoogleCloudPlatform/microservices-demo).


## Getting Started


<B>Install Dependencies</B>
```
pip install -r requirements.txt
```

<B>Run</B>

You can directly run the below commands (the default config path is `config/experiment.yaml`):
``` python
python main.py
```
Or you can modify the `config_path` in helper/paths.py.

## Parameters

The parameters in `config/experiment.yaml` are described as follows:

<B>Common args</B>
- `dataset_name`: The dataset that you want to use.
- `reconstruct`: This parameter represents whether the events should be regenerated. (Default: False)

<B>Model</B>
- `TO`: TO denotes whether the task-oriented learning module should be loaded. (Default: True)
- `CM`: CM denotes whether the cross-modal association should be established. (Default: True)
- `guide_weight`: This parameter adjusts the scale of the contrastive loss. (Default: 0.1)
- `aug`: This parameter represents whether the data augmentation strategy should be used. (Default: True)
- `aug_method`: Two data augmentation strategies are available: node_drop and random_walk. (Default: node_drop)
- `aug_percent`:  The inactivation probability. (Default: 0.2)

# **Adaptive Quantitative Trading with Imitative Recurrent Deterministic Policy Gradient**
Authors: *Lara Hofman, Mattia Martino, Sandro Mikautadze, Elio Samaha*

## Setup and code structure
This repository contains the code for the report "Adaptive Quantitative Trading with Imitative Recurrent Deterministic Policy Gradient". The required packeges are listed in the `requirements.txt` file. To install them, run the following command:
```bash
pip install -r requirements.txt
```
All the code is written in Python 3.9.21. The code is compatible with both CPU and GPU, but it is recommended to run it on a GPU for faster training and inference times. The code has been tested on a machine with an NVIDIA GeForce RTX 3090 GPU.

The code is structured as follows:
- `data/`: Contains the data used for training and testing the models.
- `models/`: Contains the checkpoints of the trained models, together with the respective losses.
- `scripts/`: Contains the definitions of the environment, and the agent, together with helper functions.
- `demo.py`: Contains the code to run the demo for the different models
- `train.py`: Contains the code to train the models

*N.B. You need a `.env` file containing your API KEY and SECRET KEY in order to download the data with our script. For privacy reasons the ones of the authors are not provided.*



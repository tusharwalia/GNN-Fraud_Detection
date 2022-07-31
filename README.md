# GNN-based-Fraud-Detection
As the internet services have expanded over the past few years, various kind of
fraudulent activities have increased manifold. These fraudsters bypass the anti-
fraud system by disguising as regular users to disperse disinformation. In order
to detect these activities many graph based solutions have performed
effectively.
Graph based methods connect entities having different relations and helps in
finding out the suspiciousness of these entities.


## Setup

You can download the project and install the required packages using the following commands:

```bash
git clone https://github.com/tusharwalia/GNN-based-Fraud-Detection.git
cd Fraud-Detection
pip3 install -r requirements.txt
```

To run the code, you need to have at least **Python 3.6** or later versions. 

## Running

1. In Fraud-Detection directory, run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets; 
2. Run `python data_process.py` to generate adjacency lists;
3. Run `python train.py` to run code with default settings.

For other dataset and parameter settings, please refer to the arg parser in `train.py`. Our model supports both CPU and GPU mode.

## Running on your datasets

To run this code on your datasets, you need to prepare the following data:

- Multiple-single relation graphs with the same nodes where each graph is stored in `scipy.sparse` matrix format, you can use `sparse_to_adjlist()` in `utils.py` to transfer the sparse matrix into adjacency lists;
- A numpy array with node labels. Currently, only supports binary classification;
- A node feature matrix stored in `scipy.sparse` matrix format. 

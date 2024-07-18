# MAN-SF - Stock Forecasting with Social Media Data and Attention Mechanism.
 
[Reproduction of the paper "Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations"](https://aclanthology.org/2020.emnlp-main.676.pdf)

We provide an alternative implementation of the model, and all the resources to run it, including how to extract and process wikirelations data.

This code uses the [stokcnet dataset](https://github.com/yumoxu/stocknet-dataset)

## Run
- Data preprocessing: `python data_preprocess.py` `python wikirelations.py`
- Then, build input data for model, `python build_intput_data.py`
- Finally, you can run it by `python train.py`
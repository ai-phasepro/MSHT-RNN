# MSHT-RNN

## Files

The **processed_dataset** folder contains the dataset used, where nonnuclear.csv is the nonnuclear protein data and nuclear.csv is the nuclear protein data.
**Bilstm_head_tail.py** is the amino acid sequence head and tail feature extraction and detection model.
**Bilstm_multi_scale_head.py** is amino acid sequence multi-scale head feature extraction and detection model.
**Bilstm_multi_scale_head_tail.py** is an amino acid sequence multi-scale head and tail feature extraction and detection model.
**Bilstm_multi_scale_tail.py** is a multi-scale tail feature extraction and detection model for amino acid sequences.

## Requirement

This code was implemented using the Pytorch framework (version 2.1.2). More details have been described in the file “requirements.txt”.

```python
pip install -r requirements.txt
```

## Run Code

### Install Requirement

Code run with python=3.8&torch=2.1.2+cu118

```python
pip install -r requirements.txt
```

### Run our model

```python
mkdir ./output_dir/Bilstm_MS_output
# multi_scale_head_tail
python Bilstm_multi_scale_head_tail.py
# multi_scale_head
python Bilstm_multi_scale_head.py
# multi_scale_tail
python Bilstm_multi_scale_tail.py
# Bilstm_head_tail
python Bilstm_head_tail.py
```

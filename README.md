# MFA_cplx
Python implementation of a complex-valued version of the expectation-maximization (EM) algorithm for fitting Mixture of Factor Analyzers (MFA). 
Some parts of the implementation are based on the PYPI implementation from
https://pypi.org/project/mofa/
with substantial speed-ups, new functionalities, and a complex-valued extension.
The EM algorithm maximizes the likelihood of a circularly symmetric Gaussian distribution.

## Instructions
The main implementation is contained in `MFA_cplx.py` with the class `MFA_cplx`.
The file `MFA_cplx_example.py` provides useful examples of how to use the code.

## Requirements
This code is written in *Python*. It uses the *numpy*, *scipy*, *sklearn*, and *time* packages. The code was tested with Python 3.7.

## Methods of `MFA_cplx`
- `fit(data)`: Fitting the MFA parameters to the provided complex-valued dataset of shape `(n_samples, n_dim)`.
  
- `predict_proba_max(data)`: Predict the labels for the data samples using trained model.

- `predict_proba(X)`: Predict posterior probability of each component given the data.

- `sample(n_samples)`: Generate random samples from the fitted MFA.


## Research work
The results of the following work are based (in parts) on the complex-valued MFA implementation:
- B. Fesl, N. Turan, and W. Utschick, “Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers,” in *57th Asilomar Conf. Signals, Syst., Comput.*, 2023.

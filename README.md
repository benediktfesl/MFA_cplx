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

## Original License
The original code from https://pypi.org/project/mofa/ is covered by the following license:

> Copyright 2012 Ross Fadely, Daniel Foreman-Mackey, David W. Hogg, and
> contributors.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in
> the Software without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
> the Software, and to permit persons to whom the Software is furnished to do so,
> subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
> FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
> IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
> CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

## Licence of Contributions
The contributions and extensions are covered by the BSD 3-Clause License:

> BSD 3-Clause License
>
> Copyright (c) 2023 Benedikt Fesl.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


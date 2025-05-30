# STD-FD: Spatio-Temporal Distribution Fitting Deviation for AIGC Forgery Identification
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/PyTorch-1.13-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)
> Abstract: With the rapid development of AIGC technologies, particularly diffusion models, the creation of highly realistic fake images that can deceive human visual perception has become possible. Consequently, various forgery identification techniques have been proposed to identify such fake content. However, existing detection approaches for AIGC-generated images remain superficial. They treat the generation process of fake images as an auxiliary tool for forgery identification or even regard it as a black-box operation, without deeply exploring the identification information from the perspective of the generative mechanism of fake data. In this paper, we propose Spatio-Temporal Distribution Fitting Deviation for AIGC Forgery Identification (STD-FD), which delves into the underlying mechanisms of the AIGC-generated process. By employing the step-by-step decomposition and reconstruction of data in generative diffusion models, initial exploratory experiments indicate that temporal distribution fitting deviations occur during the image reconstruction process. Thus, we model these deviations using reconstruction noise maps for each spatial semantic unit, derived from a super-resolution algorithm. Critical discriminant patterns, termed DFactors, are identified through the statistical modeling of these deviations. Ultimately, DFactors enhance the classifier’s accuracy in identifying forged images. Extensive experiments demonstrate that the proposed STD-FD effectively captures distribution patterns in AIGC-generated data, exhibiting strong robustness and generalizability, and outperforms existing state-of-the-art methods across major datasets in the field.
<p align="center"> 
<img src="main.jpg">
</p>

## First stage: spatial-temporal information capture
In the first stage, temporal change information for each semantic block, corresponding to the spatial superpixel, is firstly extracted (a). Then, we focus on extracting the 𝐷𝐹𝑎𝑐𝑡𝑜𝑟, which can distinguish between positive and negative samples during the sampling process. Next, global classification discriminative factors 𝐷𝐹𝑎𝑐𝑡𝑜𝑟 are constructed by identifying key changing segments that differentiate between the two types of samples (b).

- Superpixel slicing algorithm for configuring temporal noise graph data
```
python SLIC/SLIC.py # Change the file path to the actual path
```
- Generating time series data of noise maps during diffusion sampling
```
python Time_series/extract_time.py # The parameters are set to default values, and if necessary, modify the configuration parameters
```

- Generating 𝐷𝐹𝑎𝑐𝑡𝑜𝑟
```
DFactor_global/Run_learn_main.py # The parameters are set to default values, and if necessary, modify the configuration parameters
```
## Second stage: discrepancy detection via distribution fitting deviation
Based on the 𝐷𝐹𝑎𝑐𝑡𝑜𝑟 extracted from the first stage, we perform distribution fitting deviation modeling on the data to be identified. Feature engineering is completed using distance, correlation, and matching metrics (c). With the extracted
feature A𝑘, a classifier is trained for forgery identification.

- Training and Testing
```
Train/Run_shell.py # The parameters are set to default values, and if necessary, modify the configuration parameters. run.py inherits the training and testing process
```

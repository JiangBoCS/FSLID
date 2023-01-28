# FSLID

## Abstract
Deep Neural Networks (DNNs) have achieved impressive results on the task of image denoising, but there are two serious problems. First, the denoising ability of DNNs-based image denoising models using traditional training strategies heavily relies on extensive training on clean-noise image pairs. Second, image denoising models based on DNNs usually have large parameters and high computational complexity. To address these issues, this paper proposes a two-stage Few-Shot Learning for Image Denoising (FSLID). Our FSLID is a two-stage denoising strategy integrating Basic Feature Learner (BFL), Denoising Feature Inducer (DFI), and Shared Image Reconstructor (SIR). BFL and SIR are first jointly unsupervised to train on the base image dataset $\mathcal{D}_{base}$ consisting of easily collected high-quality clean images. Following this, the trained BFL extracts the guided features and constraint features for the noisy and corresponding clean images in the novel image dataset $\mathcal{D}_{novel}$, respectively. Furthermore, DFI encodes the noisy features of the noisy images in $\mathcal{D}_{novel}$. Then, inducing both the guided features and noisy features, DFI can generate the denoising prior features for the SIR with frozen weights to adaptively denoise the noisy images. Furthermore, we propose refined, low-channel-count, recursive multi-branch Multi-Scale Feature Recursive (MSFR) to modularly formulate an efficient DFI to capture more diverse contextual features information under a limited number of feature channels. Thus, the proposed MSFR can significantly reduce the number of model parameters and computational complexity. Extensive experimental results demonstrate our FSLID significantly outperforms well-established baselines on multiple datasets and settings. We hope that our work will encourage further research to explore the field of few-shot image denoising.

#### The parameter weights of the model can be downloaded [here.](https://pan.baidu.com/s/1yeTj2GHvFus6G6KA_WgXOQ)
#### Extraction code: 46y3
### Perform Inference
```
sh test.sh
```

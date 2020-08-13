# Auto-mask Music Generative Model via EC2-VAE Disentanglement

This is the work with this paper in IEEE International Conference on Semantic Computing ICSC 2020: [link](https://arxiv.org/pdf/2002.02393.pdf)

We implement [EC2-VAE](https://github.com/cdyrhjohn/Deep-Music-Analogy-Demos) into the conditional generative model to let people generate the music melody in terms of controlling rhythm patterns and chord progressions, and even extra chord function labels.

## See repo structure:
* processed_data: processed Nottingham data in EC2-VAE latent vector sequences, due to the 100MB limits, we have some missing files [here](https://drive.google.com/drive/folders/1GJPUR1hxtIylAYutshP8z6_lc3MTFS2u?usp=sharing).
* vae: EC2-VAE model
* AmMGM_model_decode.ipynb: about how to use the trained model parameters to generate the music from train/valid/test dataset.
* model_mask_cond: conditional generative model
* train_AmMGM: training model file.
* result: the vae_nottingham_output, model_generation_out, and sample_for_presentation

We did not provide the trained parameters in github, if you want find out both AmMGM-parameters and EC2-VAE-parameters we trained for this model, check out the [link](https://drive.google.com/drive/folders/1GJPUR1hxtIylAYutshP8z6_lc3MTFS2u?usp=sharing) here.

## Credit
Please cite this paper if you want to base on this work to make improvements or further research.

> @inproceedings{amg-ec2vae-icsc, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; author    = {Ke Chen and Gus Xia and Shlomo Dubnov}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; title     = {Continuous Melody Generation via Disentangled Short-Term Representations and Structural Conditions}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; booktitle = {{IEEE} 14th International Conference on Semantic Computing, {ICSC}}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pages     = {128--135}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; publisher = {{IEEE}}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; year      = {2020}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; address   = {San Diego, CA, USA} <br>
> }


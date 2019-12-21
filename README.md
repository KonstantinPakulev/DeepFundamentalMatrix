# DeepFundamentalMatrix
To reproduce the results please do the following:
1) Download South Building dataset from here https://colmap.github.io/datasets.html and convert its data using COLMAP to binary format.

2) Choose the checkpoint you want to use: [checkpoints_sh](checkpoints_sh) or [checkpoints_md](checkpoints_md) the first one is both for visual and qualitative evaluation, but provides slightly worse results. The second one is for qualitative evalution only and will require you to change some code in order to make it work.

3) If you have chosen the first checkpoint then simply run [test.sh](test.sh) by providing dataset root and model checkpoint path for qualitative evaluation. 
If you want to reproduce visual evaluation then download neural network for keypoints detection/description from here https://drive.google.com/open?id=1pKvdfSXs5al3ESbPTQaTf39xAYProvsa. Then open [notebooks/evaluation.ipybn](notebooks/evaluation.ipybn) and provide checkpoint paths for both models (i.e. detection-description network and IRLS-NN).

4) If you have chosen the second checkpoint, then you only have an option of running a qualitative evaluation. Procedure is the same as for the first checkpoint, however you will need to replace in [source/nn/model.py](source/nn/model.py) the following line 

```
vectors_init = torch.cat(((kp1[:, :, :2] + 1) / 2, (kp2[:, :, :2] + 1) / 2), 2).permute(0, 2, 1)
```

with 

```
vectors_init = torch.cat(((kp1[:, :, :2] + 1) / 2, (kp2[:, :, :2] + 1) / 2, additional_info), 2).permute(0, 2, 1)
```


# Imitation Learning with VLN  

Config file to use:  
- `habitat_baselines/config/vln/il_vln.yaml`. DAgger, teacher forcing, and student forcing all can be configured with the `P` parameter in this config.  
- `habitat_baselines/config/test/il_vln_test.yaml` for a quick test of IL training.  

Required downloads:
- `data/dd-ppo-weights/gibson-2plus-resnet50.pth` (https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth)
- `data/datasets/vln/mp3d/r2r/v1/preprocessed` (link on Slack)

Where `preprocessed` has:  
- `train`  
- `val_seen`  
- `val_unseen`  
- `embeddings.json` which are GloVe embeddings for just the relevant words in our vocab  

Each dataset split folder has:
- `{split}.json.gz` containing episode definitions and vocabulary  
- `{split}_gt.json.gz` containing ground truth actions used by DTW measures  

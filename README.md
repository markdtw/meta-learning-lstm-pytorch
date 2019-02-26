# Optimization as a Model for Few-shot Learning
Pytorch implementation of [Optimization as a Model for Few-shot Learning](https://openreview.net/forum?id=rJY0-Kcll) in ICLR 2017 (Oral)

![Model Architecture](https://i.imgur.com/lydKeUc.png)

## Prerequisites
- python 3+
- pytorch 0.4+ (developed on 1.0.1 with cuda 9.0)
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [tqdm](https://tqdm.github.io/) (a nice progress bar)

## Data
- Mini-Imagenet as described [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet)
  - You can download it from [here](https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR/view?usp=sharing) (~2.7GB, google drive link)

## Preparation
- Make sure Mini-Imagenet is split properly. For example:
  ```
  - data/
    - miniImagenet/
      - train/
        - n01532829/
          - n0153282900000005.jpg
          - ...
        - n01558993/
        - ...
      - val/
        - n01855672/
        - ...
      - test/
        - ...
  - main.py
  - ...
  ```
  - It'd be set if you download and extract Mini-Imagenet from the link above
- Check out `scripts/train_5s_5c.sh`, make sure `--data-root` is properly set

## Run
For 5-shot, 5-class training, run
```bash
bash scripts/train_5s_5c.sh
```
Hyper-parameters are referred to the [author's repo](https://github.com/twitter/meta-learning-lstm)

## Notes
- Results (This repo is developed following the [pytorch reproducibility guideline](https://pytorch.org/docs/stable/notes/randomness.html)):

|seed|train episodes|val episodes|val acc mean|val acc std|test episodes|test acc mean|test acc std|
|-|-|-|-|-|-|-|-|
|719|41000|100|59.08|9.9|100|56.59|8.4|
|  -|    -|  -|    -|  -|250|57.85|8.6|
|  -|    -|  -|    -|  -|600|57.76|8.6|
| 53|44000|100|58.04|9.1|100|57.85|7.7|
|  -|    -|  -|    -|  -|250|57.83|8.3|
|  -|    -|  -|    -|  -|600|58.14|8.5|

- The results I get from directly running the author's repo can be found [here](https://i.imgur.com/rtagm2c.png), I have slightly better performance (~5%) but neither results match the number in the paper (60%).
- The implementation replicates two learners similar to original repo:
  - `learner_w_grad` functions as a regular model, get gradients and loss as inputs to meta learner.
  - `learner_wo_grad` constructs the graph for meta learner:
    - All the parameters in `learner_wo_grad` are replaced by `cI` output by meta learner.
    - `nn.Parameters` in this model are casted to `torch.Tensor` to connect the graph to meta learner.
- Several ways to **copy** a parameters from meta learner to learner depends on the scenario:
  - `copy_flat_params`: we only need the parameter values and keep the original `grad_fn`.
  - `transfer_params`: we want the values as well as the `grad_fn` (from `cI` to `learner_wo_grad`).
    - `.data.copy_` v.s. `clone()` -> the latter retains all the properties of a tensor including `grad_fn`.
    - To maintain the batch statistics, `load_state_dict` is used (from `learner_w_grad` to `learner_wo_grad`).

## References
- [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) (Data loader)
- [pytorch-meta-optimizer](https://github.com/ikostrikov/pytorch-meta-optimizer) (Casting `nn.Parameters` to `torch.Tensor` inspired from here)
- [meta-learning-lstm](https://github.com/twitter/meta-learning-lstm) (Author's repo in Lua Torch)


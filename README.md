# Training Confidence-Calibrated Classifier for Detecting Out-of-Distribution Samples

This code is for the paper "Training Confidence-Calibrated Classifier for Detecting Out-of-Distribution Samples"

## Preliminaries

It is tested under Ubuntu Linux 16.04.1 and Python 2.7 environment, and requries following Python packages to be installed:

* [Pytorch](http://pytorch.org/): Only GPU version is available.

## Example scripts

* [`run_CMCL.sh`](run_CMCL.sh): train the models using "Confident multiple choice learning".
* [`run_MCL.sh`](run_MCL.sh): train the models using "Multiple choice learning".
* [`run_IE.sh`](run_IE.sh): train the models using "Independent ensemble".

## All training options:

    python src/ensemble.py \
    --dataset=cifar \
    --model_type=resnet \
    --batch_size=128 \
    --num_model=5 \
    --loss_type=cmcl_v0 \
    --k=4 \
    --beta=0.75 \
    --feature_sharing=True \
    --test=False

* `dataset`         : supports `cifar` and `svhn`.
* `model_type`      : supports `vggnet`, `googlenet`, and `resnet`.
* `batch_size`      : we use batch size 128.
* `num_model`       : number of models to ensemble.
* `loss_type`       : supports `independent`, `mcl`, `cmcl_v0`, and `cmcl_v1`.
* `k`               : overlap parameter.
* `beta`            : penalty parameter.
* `feature_sharing` : use feature sharing if `True`.
* `test`            : if `True`, test the result of previous training, otherwise run a new training.

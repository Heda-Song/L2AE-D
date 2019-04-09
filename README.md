# L2AE

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* TensorFlow v1.0+

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in `data/omniglot_resized/resize_images.py` and `data/miniImagenet/proc_images.py` respectively.

### Usage

### Training on Omniglot
# 5-way 1-shot
python main.py --datasource=omniglot --metatrain_iterations=70000 --meta_batch_size=4 --num_classes=5 --K_shot=1 --num_query=5 --num_query_val=5 --logdir=logs/omniglot5way1shot/
# 5-way 5-shot
python main.py --datasource=omniglot --metatrain_iterations=70000 --meta_batch_size=4 --num_classes=5 --K_shot=5 --num_query=5 --num_query_val=5 --logdir=logs/omniglot5way5shot/
# 20-way 1-shot
python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=4 --num_classes=20 --K_shot=1 --num_query=5 --num_query_val=5 --logdir=logs/omniglot20way1shot/
# 20-way 5-shot
python main.py --datasource=omniglot --metatrain_iterations=25000 --meta_batch_size=4 --num_classes=20 --K_shot=5 --num_query=5 --num_query_val=5 --logdir=logs/omniglot20way5shot/

### Training on miniImageNet
# 5-way 1-shot
python main.py --datasource=miniimagenet --metatrain_iterations=70000 --meta_batch_size=4 --num_classes=5 --K_shot=1 --num_query=15 --num_query_val=15 --logdir=logs/miniimagenet5way1shot/
# 5-way 5-shot
python main.py --datasource=miniimagenet --metatrain_iterations=70000 --meta_batch_size=4 --num_classes=5 --K_shot=5 --num_query=15 --num_query_val=15 --logdir=logs/miniimagenet5way5shot/

### Testing
To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set, and set the 'test_itr' to restore a saved model.

### Reference
[MAML](https://github.com/cbfinn/maml)
# Assignment 2: Finetuning Pre-Trained BERT for Named Entity Recognition

## Task
In this assignment, you are asked to complete the code for finetuning a [distilled BERT](https://arxiv.org/abs/1910.01108) model for the [named entity recognition (NER)](https://cs230.stanford.edu/blog/namedentity/) task on a small subset of CoNLL 2003, a famous and widely-used dataset for NER.
The training framework and auxiliary functions are already provided for you.
What you need to do is complete the code within
```python
# --- TODO: start of your code ---

# --- TODO: end of your code ---
```
so that the model can run properly.
Changing the code outside the `TODO` block is not necessary.

The functions to be completed are
- `src.dataset.dataset.Dataset.encode`: for data pro-processing.
It encodes the input tokens into the BERT token format
- `src.dataset.collate.DataCollator.__call__`: for batch processing.
It collates the instances in each sampled batch so that they meet the BERT model input format.
- `src.train.Trainer.training_step`: for model finetuning.
The part to be completed concerns the weight updating and loss tracking.
- `src.train.Trainer.get_loss`: for loss calculation.
- `src.train.Trainer.evaluate`: for model evaluatin.
The part to be completed concerns model inference.

For detailed description and requirment, please check the annotation before each `TODO` block.

## Environment Setup
The code is built on Python 3.10 and [Hugging Face Transformers Library](https://github.com/huggingface/transformers) with customized data processor and trainer.
Other package requirements are listed in `requirements.txt`.
You are suggested to run the code in an isolated virtual [conda](https://www.anaconda.com/) environment.
Suppose you have already install conda in your device, you can create a new environment and activate it with
```bash
conda create -n 310 python=3.10
conda activate 310
```
Then, you can install the required packages with
```bash
pip install -r requirements.txt
```

Alternatively, you can also use other Python version manager or virtual environments such as [pyenv](https://github.com/pyenv/pyenv) or [docker](https://www.docker.com/) to you prefer.

## Data
The dataset used for this assignment is a subset of [CoNLL 2003](https://aclanthology.org/W03-0419.pdf), the most famous benchmark dataset for NER.
For this assignment, we subsampled 1000 training data points, and 100 points for validation and test.
The data is already pre-processed for you and stored in the `./data/` directory as `.json` files.

## Run

If you are using a Unix-like system such as Linux or MacOS, you can run the code through the provided `run.sh` file.
You first need to edit `run.sh` to complete your name and GTID, and then run
```bash
./run.sh [GPU ID]
```
for example, 
```bash
./run.sh 0
```
if you want to GPU-0 to accelerate your training.
If you leave `GPU ID` blank, the model will be trained on CPU.

~~For MacOS running on M* chips, running `bash ./run.sh` will automatically take advantage of [mps accelaration](https://developer.apple.com/metal/pytorch/). You can disable this behavior by adding `--no_mps` argument into the Python call in the `sh` file.~~
This feature is deprecated as mps sometimes returns incorrect results.

Alternately, you can also run the code with 
```bash
[CUDA_VISIBLE_DEVICES=...] python run.py --name <your name> --gtid <your GTID> [other arguments...]

# for example, python run.py --name "George Burdell" --gtid 123456789 --lr 1e5 --batch_size 4096 --n_epochs 4096
```

## Submission

If your code runs successfully, you will see a `record.log` file in your `log` folder suppose you keep the `--log_path` argument as default.
The log file should track your training status, reporting the training loss, and the validation performance for each training epoch, and the final test performance.
If your code is correct, the test F1 score should be around 88 when the model is trained for 20 epochs.
You may also fine-tune the hyperparameters to achieve better performance.

For this assignment, you should submit a `ner.<GivenName>.<FamilyName>.<GTID>.zip` (e.g. `ner.George.Burdell.123456789.zip`) file containing `./src/` and `./log/` folders and all their contents.
**Do not include `./data/`, `.gitignore`, `LICENSE` or other files or folders.**
If your using Unix-like systems, you can run
```bash
zip -r ner.<GivenName>.<FamilyName>.<GTID>.zip log/ src/
```
The final `zip` file should not be larger than 50KB.

## Reference

With default hyper-parameters, each training epoch takes roughly 30s to run on Mac M1 CPU, ~~18s on mps,~~ and 2s on Nvidia A5000/RTX4090 GPU.

# Semantic Grouping Network for Video Captioning
Hobin Ryu, Sunghun Kang, Haeyong Kang, and Chang D. Yoo. AAAI 2021.
[[arxiv]](https://arxiv.org/abs/2102.00831)

# Environment

* Ubuntu 16.04
* CUDA 9.2
* cuDNN 7.4.2
* Java 8
* Python 2.7.12
  * PyTorch 1.1.0
  * Other python packages specified in requirements.txt


# Usage

### 1. Setup
   ```
   $ pip install -r requirements.txt
   ```

### 2. Prepare Data
   1. Extract features from datasets and locate them at `data/<DATASET>/features/<NETWORK>.hdf5`.
   
      > e.g. ResNet101 features of the MSVD dataset will be located at `data/MSVD/features/ResNet101.hdf5`.
   
      > I refer to [this repo](https://github.com/hobincar/pytorch-video-feature-extractor) for extracting the ResNet101 features, and [this repo](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) for extracting the 3D-ResNext101 features.

   2. Split the features into train, val, and test sets by running following commands.
      ```
      $ python -m split.MSVD
      $ python -m split.MSR-VTT
      ```

### 3. Prepare The Code for Evaluation
   Clone the evaluation code from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).
   ```
   $ git clone https://github.com/tylin/coco-caption.git
   $ mv coco-caption/pycocoevalcap .
   $ rm -rf coco-caption
   ```

### 4. Extract Negative Videos
   ```
   $ python extract_negative_videos.py
   ```
   or you can skip this step as the output files are already uploaded at `data/<DATASET>/metadata/neg_vids_<SPLIT>.json`

### 5. Train
   ```
   $ python train.py
   ```
   You can change some hyperparameters by modifying `config.py`.

### 6. Evaluate
   ```
   $ python evaluate.py --ckpt_fpath <MODEL_CHECKPOINT_PATH>
   ```

# License
The source-code in this repository is released under MIT License.

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
   1. Download the GloVe Embedding from [here](https://drive.google.com/file/d/12l40qeIi5zioX2WaVBLXD_oarc8zbaqG/view?usp=sharing) and locate it at `data/Embeddings/GloVe/GloVe_300.json`.
   2. Extract features from datasets and locate them at `data/<DATASET>/features/<NETWORK>.hdf5`.
   
      > e.g. ResNet101 features of the MSVD dataset will be located at `data/MSVD/features/ResNet101.hdf5`.
   
      > I refer to [this repo](https://github.com/hobincar/pytorch-video-feature-extractor) for extracting the ResNet101 features, and [this repo](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) for extracting the 3D-ResNext101 features.

   3. Split the features into train, val, and test sets by running following commands.
      ```
      $ python -m split.MSVD
      $ python -m split.MSR-VTT
      ```
   *You can skip step 2-3 and download below files*
   * MSVD
     - ResNet-101 [[train]](https://drive.google.com/file/d/1dRg6cfee92tnulT6syPpt1a696ZTYPwb/view?usp=sharing)
                  [[val]](https://drive.google.com/file/d/1g_uXfrr41inUy92Ez44wiNv5ZVIGT5l5/view?usp=sharing)
                  [[test]](https://drive.google.com/file/d/11GsImQ8vhu1HpkQx4XnVGiM6r-enc2aP/view?usp=sharing)
     - 3D-ResNext-101 [[train]](https://drive.google.com/file/d/1-o-KQRXq-ICjDFSmyj-tyC4a-BcUW6VD/view?usp=sharing)
                      [[val]](https://drive.google.com/file/d/1jPc0zsv3kGukV8KtuJXs4mv4oMRLDUku/view?usp=sharing)
                      [[test]](https://drive.google.com/file/d/1dklmabW4CdjSCH6um-Yu8WjgeGPwwNla/view?usp=sharing)
   * MSR-VTT
     - ResNet-101 [[train]](https://drive.google.com/file/d/1C_DXGOXIIgvgoBog1pwejTOAeW0HX4rW/view?usp=sharing)
                  [[val]](https://drive.google.com/file/d/10ZpgO-LTdxwQNyKDudd1yRVZvv2HQty-/view?usp=sharing)
                  [[test]](https://drive.google.com/file/d/1YletZy4YVLkM_lnF4zzvinAswlo5dIPM/view?usp=sharing)
     - 3D-ResNext-101 [[train]](https://drive.google.com/file/d/1ieGl5eB4LwP90gQcpztwlhWMsbNURtm-/view?usp=sharing)
                      [[val]](https://drive.google.com/file/d/10zkim64Uk9yptPmBX3tky-CJy4z-Njl5/view?usp=sharing)
                      [[test]](https://drive.google.com/file/d/1jmXa8Lf9vIFK1_nQzQ1mbpyxums7ZyXY/view?usp=sharing)
                      
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
   
*Pretrained Models - SGN(R101+RN)*
* MSVD: https://drive.google.com/file/d/12Xjd8VdDiyvBxM9sPnnXz87Wa_eVv0ii/view?usp=sharing
* MSR-VTT: https://drive.google.com/file/d/1kx7FBi2UBCgIP7R9ideMpwXY0Gnqn7Yx/view?usp=sharing

*\*Disclaimer: The models above do not have the same weight as the models used in the paper (I trained them again because I lost).*

### 6. Evaluate
   ```
   $ python evaluate.py --ckpt_fpath <MODEL_CHECKPOINT_PATH>
   ```

# License
The source-code in this repository is released under MIT License.

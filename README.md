# SCGaussian
SCGaussian is a structure consistent Gaussian Splatting method which leverages the matching prior to learn 3D consistent scene structure. SCGaussian optimizes the scene structure in two folds: rendering geometry and, more importantly, the position of Gaussian primitives, which is hard to be directly constrained in the vanilla 3DGS due to the non-structure property. To ahcieve this, SCGaussian presents a hybrid Gaussian representation consisting of ray-based Gaussian primitives and ordinary non-structure Gaussian primitives. Details are described in our paper:
> Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis
>
> Rui Peng, Wangze Xu, Luyang Tang, Liwei Liao, Jianbo Jiao, Ronggang Wang
>
> NeurIPS 2024 ([arxiv](https://arxiv.org/abs/2411.03637))

<p align="center">
    <img src="./.github/images/sample.gif" width="100%"/>
</p>

📍 If there are any bugs in our code, please feel free to raise your issues.

## ⚙ Setup
#### 1. Recommended environment
```
git clone https://github.com/prstrive/SCGaussian.git


conda env create --file environment.yml
conda activate scgaussian

git clone https://github.com/ashawkey/diff-gaussian-rasterization --recursive
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install submodules/diff-gaussian-rasterization submodules/simple-knn

```

#### 2. Dataset Download

**LLFF dataset**

Download LLFF dataset from the official [download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

**DTU Dataset**

Download DTU dataset from the [official website](https://roboimagedata.compute.dtu.dk/?page_id=36/) and download the mask for evaluation from [this link](https://drive.google.com/file/d/1Yt5T3LJ9DZDiHbtd9PDFNHqJAd7wt-_E/view?usp=sharing).

**Note:** for DTU dataset, you need first run `convert.py` to get the poses and the undistorted images.

#### 3. Match Prior Extraction

We use GIM in our paper. Clone the GIM model from [here](https://github.com/xuelunshen/gim) and setup the enviroment following their instructions. We provide a script [here](data_preprocess/get_match_info.py) to extract match prior, and you need to specify your own data path first.

We provide our pre-extracted LLFF match priors [here](https://drive.google.com/drive/folders/1ZT9fal1uQaqUxLur3XIRtgqNvZD0H9VK?usp=sharing), and move the `match_data.npy` to the corresponding scene folder. The final data structure maybe likes this:
```
LLFF
  ├── fern                          
      ├── images
      ├── sparse
      └── match_data.npy
  ├── ...
```

## 🚀 Evaluation

#### 1. Optimization

Optimize the model for the specific scene first:
```
python train.py -s <path to scene> -m <path to save outputs> -r 8 --eval
```

#### 2. Rendering

Then render the novel view synthesis results:
```
python render.py -m <path to save outputs>
```

We also privide the video rendering script, which interpolates the camera motion withine the input camera poses:
```
python render_video.py -m <path to save outputs>
```

#### 3. Metrics

Compute the quantitative results:
```
python metrics.py -m <path to save outputs>
```

## ⏳ Customized Dataset

Our method is developded on 3DGS, and it can read and process your customized dataset using the same procedure declained in [3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting), e.g., COLMAP format datasets.

## ⚖ Citation
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{peng2024eccv,
  title={Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis},
  author={Peng, Rui and Xu, Wangze and Tang, Luyang and Liao, Liwei and Jiao, Jianbo and Wang, Ronggang},
  booktitle={Thirty-Eighth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

```

## 👩‍ Acknowledgements

This code is developed on [gaussian_splatting](https://github.com/graphdeco-inria/gaussian-splatting), we use the pretrained matching model [GIM](https://github.com/xuelunshen/gim), and we use the [modified rasterization](https://github.com/ashawkey/diff-gaussian-rasterization) to render depth.


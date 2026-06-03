# Ref-DGS:Reflective Dual Gaussian Splatting

## [Project Page](https://njfan.github.io/Ref-DGS/) | [Paper](https://arxiv.org/pdf/2603.07664) | [arXiv](https://arxiv.org/abs/2603.07664)

> Ref-DGS:Reflective Dual Gaussian Splatting<br>
> [Ningjing Fan](https://github.com/njfan), [Yiqun Wang](https://sites.google.com/view/yiqun-wang/home), [Dong-Ming Yan](https://scholar.google.com/citations?user=xAFSO3AAAAAJ&hl=en&oi=sra), [Peter Wonka](https://peterwonka.net/)<br>
> SIGGRAPH 2026

![teaser](assets/teaser.png)

## ⚙️ Setup

```bash
conda create -n refdgs python=3.11
conda activate refdgs
# Install PyTorch according to your CUDA version. We use **CUDA 11.8** as an example.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

pip install submodules/diff-surfel-2dgs --no-build-isolation
pip install submodules/diff-surfel-rasterization --no-build-isolation
pip install submodules/diff-surfel-rasterization-feature --no-build-isolation
pip install submodules/diff-surfel-rasterization-real --no-build-isolation
pip install submodules/fused-ssim --no-build-isolation
pip install submodules/simple-knn --no-build-isolation

git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast --no-build-isolation
```

## 📦 Dataset
We mainly test our method on [ShinySynthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip), [RefReal](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), [GlossySynthetic](https://liuyuan-pal.github.io/NeRO/) and [GlossyReal](https://liuyuan-pal.github.io/NeRO/). Please run the script `nero2blender.py` to convert the format of the Glossy Synthetic dataset.

Put them under the `data` folder:
```bash
data
└── Ref-NeRF
    └── refnerf
        └── car
        └── toaster
    └── ref_real
        └── gardenspheres
└── Glossy
    └── GlossySynthetic
        └── angel_blender
        └── bell_blender
    └── GlossyReal
        └── bear
```

## 🧭 Depth and Normal Priors

Pre-computed depth and normal priors are available on [Hugging Face](https://huggingface.co/datasets/njfan/Ref-DGS_Priors). Download and place them at the project root:

```bash
git clone https://huggingface.co/datasets/njfan/Ref-DGS_Priors
mv Ref-DGS_Priors priors
```

For the synthetic datasets **ShinySynthetic** and **GlossySynthetic**, depth and normal priors are inferred by [VGGT](https://github.com/facebookresearch/vggt).  
For the real-world datasets **RefReal** and **GlossyReal**, depth and normal priors are inferred by [Metric3D](https://github.com/YvanYin/Metric3D).

The expected directory structure is:

```bash
priors
└── Ref-NeRF
    └── refnerf
        └── scene_name
            └── depth
                └── <image_name>.pth
            └── normal
                └── <image_name>.png
    └── ref_real
        └── scene_name
            └── depth
                └── <image_name>.pth
            └── normal
                └── <image_name>.jpg
└── Glossy
    └── GlossySynthetic
        └── scene_name
            └── depth
            └── normal
    └── GlossyReal
        └── scene_name
            └── depth
            └── normal
```


## 🏃 Training / Testing / Evaluation

We provide unified scripts for each dataset:

- **ShinySynthetic**: `scripts/run_shiny.py`  
- **GlossySynthetic**: `scripts/run_glossy_syn.py`  
- **RefReal**: `scripts/run_ref_real.py`  
- **GlossyReal**: `scripts/run_glossy_real.py`

Before running, please edit in the script:
- `data_base_path` → your dataset root path  
- `gpu_id` → GPU id used for this dataset  

For synthetic datasets, we use `train.py` for training and `render.py` for rendering/testing/evaluation.  
For real datasets, we use `train-real.py` for training and `render_real.py` for rendering/testing/evaluation.


## 🫡 Acknowledgments

This work is built on many amazing research works and open-source projects,

- [3D Gaussian Splatting with Deferred Reflection](https://github.com/gapszju/3DGS-DR/tree/main)
- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)
- [Ref-GS : Directional Factorization for 2D Gaussian Splatting](https://ref-gs.github.io/)

We are grateful to the authors for releasing their code.

## 📜 Citation

If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```
@article{fan2026ref,
  title={Ref-DGS: Reflective Dual Gaussian Splatting},
  author={Fan, Ningjing and Wang, Yiqun and Yan, Dongming and Wonka, Peter},
  journal={arXiv preprint arXiv:2603.07664},
  year={2026}
}
```

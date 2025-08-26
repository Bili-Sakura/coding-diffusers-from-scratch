## Tutorial

## Tutorial 0 - Introduction and Utils

> [TODO]: Implement [utils.py](./utils.py) (utils)

## Tutorial 1 — Diffusion Transformer (DiT)

- [dit.py](./dit.py) (main)
- [pipeline_dit.py](./pipeline_dit.py) (main)
- [utils.py](./utils.py) (utils)

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/089509d1-c339-4fdb-a6f1-35a9945ab305/1b34d5107a2864e22f2cfe596414a1a4c3c8708fcecf4c6649fbecdfc4c32aef.jpg)
![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/089509d1-c339-4fdb-a6f1-35a9945ab305/329c0d85afe09b29a6e34fe5607d64a3af46f49afa92c889dcd64708d7c20f0c.jpg)

<table><tr><td>Model</td><td>Layers N</td><td>Hidden size d</td><td>Heads</td><td>Gflops (I=32, p=4)</td></tr><tr><td>DiT-S</td><td>12</td><td>384</td><td>6</td><td>1.4</td></tr><tr><td>DiT-B</td><td>12</td><td>768</td><td>12</td><td>5.6</td></tr><tr><td>DiT-L</td><td>24</td><td>1024</td><td>16</td><td>19.7</td></tr><tr><td>DiT-XL</td><td>28</td><td>1152</td><td>16</td><td>29.1</td></tr></table>

```bibtex
@InProceedings{Peebles_2023_ICCV,
  author    = {Peebles, William and Xie, Saining},
  title     = {Scalable Diffusion Models with Transformers},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {4195-4205}
}
```

## Tutorial 2 — Stable Diffusion 2

- [sd.py](./sd.py) (main)
- [pipeline_sd.py](./pipeline_sd.py) (main)
- [utils.py](./utils.py) (utils)

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-20/6feab792-a8e7-4f99-9393-c7ea4d092c0f/d07d23bc3f2615820c5e8c2f0d1bc57eb5a3abc32213c24868d705941af3790b.jpg)

```bibtex
@inproceedings{rombachHighResolutionImageSynthesis2022,
  title = {High-{{Resolution Image Synthesis With Latent Diffusion Models}}},
  booktitle = {Proceedings of the {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}}},
  author = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  year = {2022},
  pages = {10684--10695},
  urldate = {2024-06-29},
  langid = {english},
  keywords = {Stable Diffusion},
}
@inproceedings{siFreeUFreeLunch2024,
  title = {{{FreeU}}: {{Free Lunch}} in {{Diffusion U-Net}}},
  shorttitle = {{{FreeU}}},
  booktitle = {Proceedings of the {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}}},
  author = {Si, Chenyang and Huang, Ziqi and Jiang, Yuming and Liu, Ziwei},
  year = {2024},
  pages = {4733--4743},
  urldate = {2025-08-20},
  langid = {english},
}

```

## Tutorial 3 — Denoising Diffusion Probabilistic Models (DDPMs)

- [ddpm.py](./ddpm.py) (main)
- [pipeline_ddpm.py](./pipeline_ddpm.py) (main)
- [utils.py](./utils.py) (utils)

```bibtex
@inproceedings{hoDenoisingDiffusionProbabilistic2020,
  title = {Denoising {{Diffusion Probabilistic Models}}},
  booktitle = {Advances in {{Neural Information Processing Systems}}},
  author = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  year = {2020},
  volume = {33},
  pages = {6840--6851},
  publisher = {Curran Associates, Inc.},
  urldate = {2024-06-13},
}

```

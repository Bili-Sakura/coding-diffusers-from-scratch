## Tutorial

## Tutorial 0 - Introduction and Utils

> [TODO]: Implement [utils.py](./utils.py) (utils)

## Tutorial 1 — Diffusion Transformer (DiT)

- [DiT.py](./DiT.py) (main)
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

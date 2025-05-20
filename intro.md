# Introduction

Nested Sampling is a particle Monte Carlo algorithm that has seen widespread usage in the physical sciences. Its popular implementations have often been performed in bespoke packages, which hinders wider adoption and generic comparison.

The work of {cite}`yallup2025nested` presented the atomic components of the Nested Sampling paradigm in the style of the popular `jax` based sampling library `blackjax`. This has a number of benefits, including:
- Compatibility of the atomic components with modern python PPLs such as [numpyro](https://num.pyro.ai/en/latest/index.html#)
- Clear separation of design choices from core algorithm, allowing advanced experimentation with composable components
- Unique compatibility with natively vectorized likelihood code.

Following the example of the main `blackjax` library of having a separate pedagogical sampling book, we introduce in these pages the _nested sampling book_, aiming to provide physics motivated use cases focussing on the nested sampling algorithm.

## Installation

For now the core library code is available as a fork of blackjax on the handley-lab github (https://github.com/handley-lab/blackjax), specifically the `nested_sampling` branch.

```bash
pip install git+https://github.com/handley-lab/blackjax@nested_sampling
```

All other non-standard dependencies in the examples contained in this book are listed in the notebooks themselves.

## Citation
Usage of the core algorithm should cite both the `blackjax` repo {cite}`cabezas2024blackjax`

```latex
@misc{cabezas2024blackjax,
      title={BlackJAX: Composable {B}ayesian inference in {JAX}},
      author={Alberto Cabezas and Adrien Corenflos and Junpeng Lao and Rémi Louf},
      year={2024},
      eprint={2402.10797},
      archivePrefix={arXiv},
      primaryClass={cs.MS}
}
```

as well as the pending implementation paper {cite}`yallup2025nested`

```latex
@inproceedings{
yallup2025nested,
title={Nested Slice Sampling},
author={David Yallup and Namu Kroupa and Will Handley},
booktitle={Frontiers in Probabilistic Inference: Learning meets Sampling},
year={2025},
url={https://openreview.net/forum?id=ekbkMSuPo4}
}
```

Usage of any of the physics examples should follow and include any further relevant citations detailed in the example notebooks.

## Contribution

Contributions are most welcome! Please see the [contribution guidelines](https://github.com/handley-lab/nested-sampling-book/blob/main/CONTRIBUTING.md) for more information. Or start by raising an issue on the book repository https://github.com/handley-lab/nested-sampling-book

```{tableofcontents}
```


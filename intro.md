# Introduction

Nested Sampling is a particle Monte Carlo algorithm that has seen widespread usage in the physical sciences. Its popular implementations have often been performed in bespoke packages, which hinders wider adoption and generic comparison.

The work of {cite}`Yallup2026` presented the atomic components of the Nested Sampling paradigm in the style of the popular `jax` based sampling library `blackjax`. This has a number of benefits, including:
- Compatibility of the atomic components with modern python PPLs such as [numpyro](https://num.pyro.ai/en/latest/index.html#)
- Clear separation of design choices from core algorithm, allowing advanced experimentation with composable components
- Unique compatibility with natively vectorized likelihood code.

Following the example of the main `blackjax` library, which maintains a separate pedagogical [Sampling Book](https://blackjax-devs.github.io/sampling-book/), we introduce in these pages the _nested sampling book_, aiming to provide physics motivated use cases focussing on the nested sampling algorithm.

```{note}
**Now upstreamed into BlackJAX.** The nested sampling implementation now lives in the main [BlackJAX repository](https://github.com/blackjax-devs/blackjax), in the [`blackjax.ns`](https://github.com/blackjax-devs/blackjax/tree/main/blackjax/ns) subpackage (top-level entry points `blackjax.nss` and `blackjax.nsswig`). It has not yet appeared in a tagged PyPI release, so for now it must be installed from the `main` branch with `git` — see the Installation section below.

BlackJAX's own Sampling Book contains a complementary [Nested Sampling chapter](https://blackjax-devs.github.io/sampling-book/algorithms/nested-sampling/) that introduces `blackjax.nss` and contrasts it with tempered SMC on multimodal and phase-transition targets. That page and this book are companions: it motivates the algorithm against the other samplers in BlackJAX, while these pages focus on physics-motivated use cases and the [anesthetic](https://anesthetic.readthedocs.io/) post-processing workflow.
```

## Installation

The nested sampling code is part of the main BlackJAX repository but has not yet been published in a tagged PyPI release, so for now install it directly from the `main` branch. This requires `git`:

```bash
pip install git+https://github.com/blackjax-devs/blackjax.git
```

Once a release including the `blackjax.ns` subpackage is published to PyPI, this will simplify to `pip install blackjax`.

If you have code written against the older `handley-lab/blackjax` fork (tag `v0.1.0-beta`), the [Backwards Compatibility](legacy/quickstart.ipynb) page preserves the old Quickstart and tabulates the (small) API differences.

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

as well as the implementation paper {cite}`Yallup2026`

```latex
@article{Yallup2026,
  title         = {Nested Slice Sampling: Vectorized Nested Sampling for {GPU}-Accelerated Inference},
  author        = {Yallup, David and Kroupa, Namu and Handley, Will},
  journal       = {Transactions on Machine Learning Research},
  year          = {2026},
  issn          = {2835-8856},
  url           = {https://openreview.net/forum?id=5mF2eRl3gt},
  eprint        = {2601.23252},
  archivePrefix = {arXiv},
  primaryClass  = {stat.CO},
}
```

Usage of any of the physics examples should follow and include any further relevant citations detailed in the example notebooks.

## Contribution

Contributions are most welcome! Please see the [contribution guidelines](https://github.com/handley-lab/nested-sampling-book/blob/main/CONTRIBUTING.md) for more information. Or start by raising an issue on the book repository https://github.com/handley-lab/nested-sampling-book

```{tableofcontents}
```


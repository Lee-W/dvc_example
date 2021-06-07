[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# DVC example - digit recognizer

Use the example in [Recognizing hand-written digits](https://scikit-learn.org/0.24/auto_examples/classification/plot_digits_classification.html) to demonstrate how to apply [DVC](https://dvc.org/) to existing projects.

## Getting Started

### Prerequisites
* [Python](https://www.python.org/downloads/)
* [pipx](https://pypa.github.io/pipx/)

### Set up environment

* Install `pipenv` and `invoke` through `pipx`

```sh
pipx install pipenv invoke
```

* Install dependencies through `invoke`

```sh
invoke init-dev
```

Read `tasks.py` if you want to know that's done under the hook


## Authors
Wei Lee <weilee.rx@gmail.com>

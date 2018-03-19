# PyCloud

A comparison of different cloud-based distributed and multi-processing approaches to SGD. Designed for enabling scientific researchers to fully utilize their resources quickly.

## Setup

```bash

pip install -r requirements.txt

```

## Running

Full evaluation (for local hogwild).

```bash

./evaluate.sh

```

Running one 'execution'.

```bash

python3 hogwild/main.py --epochs 3

```

## Credits

We implement and adapt methods from the following publications:

* https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf
* http://www.stat.ucdavis.edu/~chohsieh/wildSGD.pdf

We strongly rely on previous implementations:

* https://github.com/pytorch/examples/tree/master/mnist_hogwild

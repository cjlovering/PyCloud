# PyCloud

Predicates the best VM for a machine learning model workload.

## Setup

```bash

./prepare.sh

```

## Running

Evaluate on current VM.

```bash

./evaluate.sh

```

## Credits

We implement and adapt methods from the following publications:

* https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf
* http://www.stat.ucdavis.edu/~chohsieh/wildSGD.pdf

We strongly rely on previous implementations for easy reproducibility.

* https://github.com/pytorch/examples/tree/master/mnist_hogwild

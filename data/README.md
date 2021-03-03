# Datasets

To conduct the experiments, 2 datasets were used.

1. [`MalDroid2020`](https://www.unb.ca/cic/datasets/maldroid-2020.html)
2. [`AndroZoo`](https://androzoo.uni.lu/)

While `MalDroid2020` was used as a base, `AndroZoo` was used to collect new benign APKs.

## Format
This directory contains the hashes of APKs in `*.sha256` files.
Each line in the file consists of `sha256 name` pairs.
There are 4 such files

1. `train_old.sha256` - The training samples from `MalDroid2020`
2. `train_new.sha256` - The training samples from `AndroZoo`
3. `test_old.sha256` - The testing samples from `MalDroid2020`
4. `test_new.sha256` - The testing samples from `AndroZoo`
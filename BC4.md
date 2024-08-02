# BlueCrystal4 Usage Instructions

For the full documentation regarding connection, checking status of jobs, please refer to [HPC docs](https://www.acrc.bris.ac.uk/protected/hpc-docs/connecting/index.html).

If training or testing is done on BlueCrystal4, please following steps instead. Replace the account code in each script as appropriate.

## Single Parameter

- Training: Go to repository root directory and run

```bash
sbatch bc4_train.sh
```

- Synthetic Testing: Change testing image directory argument in ``bc4_synthetic_testing.sh``. Then run

```bash
sbatch bc4_testing.sh
```

- Real Testing: Change testing image directory and saved model state arguments in ``bc4_real_testing.sh``. Then run

```bash
sbatch bc4_real_testing.sh
```

## Multi Parameter

- Training: Go to repository root directory and run

```bash
sbatch bc4_multi_train.sh
```

- Synthetic Testing: Change testing image directory argument in ``bc4_synthetic_testing.sh``. Then run

```bash
sbatch bc4_multi_testing.sh
```

- Real Testing: Change testing image directory and saved model state arguments in ``bc4_real_testing.sh``. Then run

```bash
sbatch bc4_multi_real_testing.sh
```

# ImputeEHR

Explore the most effective imputation strategies for EHR data

The example dataset is [TJH dataset](https://www.nature.com/articles/s42256-020-0180-7), preprocessed following [pyehr](https://github.com/yhzhu99/pyehr) scripts.

## Imputation Methods


### Rule-based


- [ ] `LOCFImpute` (last-observation-carried-forward)
- [x] `ZeroImpute`
- [ ] `MeanImpute`
- [ ] `MedianImpute`
- [ ] `ModeImpute`
- [ ] `GRUDSimpleImpute` (GRU-D Simple, concatenating the measurement with masking and time interval)

### ML-based

- [ ] interpolation
- [ ] smoothing
- [ ] spline
- [x] k-nearest neighbor
- [ ] multiple imputation by chained equations
- [ ] random forest (e.g. MissForest)
- [ ] matrix factorization
- [ ] PCA
- [ ] SoftImpute

### DL-based

- [ ] MLP
- [ ] RNN
- [ ] AE (AutoEncoder)
- [ ] GAN

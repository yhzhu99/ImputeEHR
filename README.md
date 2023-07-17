# ImputeEHR

Explore the most effective imputation strategies for EHR data

The example dataset is [TJH dataset](https://www.nature.com/articles/s42256-020-0180-7), preprocessed following [pyehr](https://github.com/yhzhu99/pyehr) scripts.

## Imputation Methods


### Rule-based


- [x] `LOCFImpute` (last-observation-carried-forward)
- [x] `ZeroImpute`
- [x] `MeanImpute`
- [x] `MedianImpute`
- [x] `ModeImpute`
- [ ] `SimpleImpute` (GRU-D Simple, concatenating the measurement with masking and time interval)

### ML-based

- [x] `InterpolationImpute`
- [x] `SmoothingImpute`
- [x] `SplineImpute`
- [x] `KNNImpute`
- [x] `MICEImpute` (multiple imputation by chained equations)
- [x] `RFImpute` (Random Forest, e.g. MissForest)
- [x] `MFImpute` (matrix factorization)
- [x] `PCAImpute`
- [x] `SoftImpute`

### DL-based

- [ ] `MLPImpute`
- [ ] `RNNImpute`
- [ ] `AEImpute` (AutoEncoder)
- [ ] `GANImpute`

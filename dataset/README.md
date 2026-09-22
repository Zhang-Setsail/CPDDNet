# Local dataset links

The training data is not committed to this repository. Create links inside this directory:

```bash
ln -s /path/to/FinalDataHigh FinalDataHigh
ln -s /path/to/FinalDmData FinalDmData
```

`FinalDataHigh` needs `input_train/`, `input_valid/`, and `label/`. Its labels include `SceneXX.npy` clean RAW and `SceneXX_{0,45,90,135}.png` RGB images. `FinalDmData` needs `train/`, `valid/`, and `label/`. The latter provides the split and RGB labels for Stage I demosaicking. Clean RAW is read from `FinalDataHigh/label/`. The three CPDDNet training programs do not use `HighRGB2RGB`.

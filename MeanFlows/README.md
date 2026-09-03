# MeanFlow CelebA

A compact MeanFlow implementation for CelebA using PyTorch.

## Dataset layout

Set `data_root` in `config.py` to a directory containing the image files. Flat folders are supported:

```text
data/
  000001.jpg
  000002.jpg
```

The default path points to this project's `data/` directory, independent of the directory from which you start the command.

## Train

```bash
pip install -r requirements.txt
python train.py
```

For a quick smoke test, temporarily set `steps`, `sample_every`, and `ckpt_every` to `10` in `config.py`.
Resume by setting `CONFIG["resume"]` to a checkpoint path.

## Sample

Set `checkpoint`, `sample_output`, and `sample_steps` in `config.py`, then run `python sample.py`.
The command also saves `trajectory.png`, a six-frame view of one sample evolving from
Gaussian noise to the generated face. Add the generated image below after sampling:

![MeanFlow trajectory from noise to generated image](runs/celeba_meanflow/trajectory.png)

The columns show the same sample at evenly spaced integration times. The first column
is the initial noise and the last column is the final generated image.

## Evaluate

Set `checkpoint`, `eval_batches`, and `eval_output` in `config.py`, then run `python eval.py`.

The default U-Net is intended for 64x64 training. Use a batch size that fits GPU memory; MeanFlow's JVP objective uses more memory than ordinary velocity matching.

# RunPod: fine-tune exp_002 at 2048

The one experiment with measured evidence behind it and no Kaggle quota to run
it on. Inference at 2048 on 1280-trained weights already lifted validation PQ
0.3736 -> 0.4064 and the leaderboard 0.32 -> 0.33. This *trains* at 2048 rather
than merely inferring there, removing the scale mismatch instead of paying it.

## Pod

| | |
|---|---|
| GPU | **RTX 4090 24GB** (community cloud, about $0.34/h) |
| Template | RunPod PyTorch 2.x (CUDA preinstalled — do not use a bare Ubuntu image) |
| Disk | 30 GB container is enough; the dataset is 751 MB |
| Budget | $2.95 total |

Why the 4090: at roughly $0.34/h it fine-tunes 150 epochs at 2048 for about
$0.83, against $1.57 on an L40S and $2.01 on an A100. A 3090 is a few cents
cheaper per epoch but takes an hour longer, and with a fixed budget wall clock
is money whenever anything goes wrong. 24 GB holds `yolo11m-seg` at 2048; the
larger cards buy headroom this run does not need.

## Run

Set the two Kaggle variables in the pod environment, then:

```bash
git clone --depth 1 --branch claude/kaggle-credentials-setup-f7nudy \
  https://github.com/Cosmic-Witness/Sol /workspace/Sol
bash /workspace/Sol/runpod/bootstrap.sh 3.5      # hours of training
```

`bootstrap.sh` installs dependencies while downloading the data, prepares the
YOLO dataset, trains under the wall-clock budget, predicts at 2048, and
**submits to Kaggle from inside the pod**. Nothing needs retrieving afterwards,
so the pod can be terminated the moment it prints `SUBMITTED`.

## Cost

At $0.34/h with 3.5 h of training:

| stage | time | cost |
|---|---|---|
| setup, download, dataset prep | ~0.3 h | ~$0.10 |
| training | 3.5 h | ~$1.19 |
| inference over 180 images at 2048 | ~0.2 h | ~$0.07 |
| **total** | ~4.0 h | **~$1.36** |

That leaves roughly $1.59 of the $2.95 as margin for a slower pod, a retry, or a
longer run. Raise the argument to 6.0 to spend more of it; the script stops on
the clock, so the number is a spending decision rather than a guess about
convergence.

## What is worth watching

- **Batch at 2048.** 4 on a 24 GB card. On a T4 this configuration OOMed at 8
  and the large backbone could only run at 1 image per GPU, which is what forced
  the medium backbone originally. If the pod reports 48 GB the script doubles it.
- **Validation cadence.** `val_period=3` buys back two thirds of the validation
  cost as training. At 2048 that is a meaningful share of the run.
- **The exchange rate.** On this project roughly a third of a validation gain
  has reached the leaderboard (val +0.033 produced public +0.010). Expect the
  same discount here rather than reading validation PQ as a forecast.

# MicroCurr: A Tiny Curriculum-Learning Regression Model

We present MicroCurr, A minimal TensorFlow 2 script demonstrating curriculum learning with “hyper-epochs”: multiple full runs through your curriculum schedule, sinusoidal resets, and a One-Cycle learning-rate schedule.

## 🚀 Features

* **Hyper-Epochs**: Run N full curriculum cycles (hyper-epochs), each resetting your curriculum schedule.
* **Sinusoidal Curriculum**: Smooth 0→1→0 parameter ramp per hyper-epoch via a `SinusoidalCurriculum` callback.
* **One-Cycle LR Schedule**: Fast warmup + cosine-decay to stabilize training.
* **Dual Validation**: Standard core validation (`val_loss`) and “current” curriculum validation (`val_curr`) via a custom callback.
* **Lightweight**: \~100 lines of code, plug in your own data generators.

## 🔧 Prerequisites

* Python 3.8+
* [TensorFlow 2.x](https://www.tensorflow.org/)
* [TensorFlow Probability](https://www.tensorflow.org/probability)
* NumPy
* Matplotlib (for plotting)

```bash
pip install tensorflow tensorflow-probability numpy matplotlib
```

## ⚙️ Configuration

At the top of `MicroCurr-V3.py`, adjust:

```python
HYPER_EPOCHS       = 5        # full curriculum runs
EPOCHS_PER_HYPER   = 20       # epochs per run
BATCH_SIZE         = 128
STEPS_PER_EPOCH    = 200

# Curriculum end-points (e.g. max oblateness, rotation freq, sigmoid shape)
OBLATENESS_END     = 0.5
ROT_FREQ_END       = 2.0
SIGMOID_SCALE      = 0.5

# One-Cycle LR params
BASE_LR            = 1e-4
PEAK_LR            = 2.5e-4
MIN_LR             = 1e-6
PCT_WARMUP         = 0.15
```

Hook your own data-generators:

* `curriculum_generator()` → yields `(x_batch, y_batch)` following your curriculum schedule.
* `val_core_gen()` → yields core validation batches for `val_loss`.

## ▶️ Running

```bash
python MicroCurr-V3.py
```

* Training logs include `loss`, `val_loss`, and `val_curr`.
* At the end, a summary prints “floor”, “tail”, and “drift” for each metric.
* A Matplotlib plot overlays training (smoothed), core-val, and curriculum-val curves.

## 🔄 Callbacks

* **SinusoidalCurriculum**
  Resets curriculum parameters each hyper-epoch via a user-supplied `update_fn(scale)`.

* **ValCurrCB**
  Measures “current” curriculum performance every epoch using a Huber loss over a few validation steps.

## 🎨 Customization

* Swap in different curriculum schedules (linear, exponential).
* Tweak LR schedule: replace `OneCycleSchedule` with any `LearningRateSchedule`.
* Plug in more complex models or residual blocks.

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

> Feel free to raise issues or PRs for new features, fixes, or enhancements!

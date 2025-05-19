# MicroCurr: A Tiny Curriculum-Learning Regression Model

We present MicroCurr, a single-file TensorFlow 2 demo. It combines curriculum learning with "hyper-epochs" (multiple full curriculum cycles), built‑in streaming data generation, and an integrated plotting utility — all out of the box.

## 🚀 Features

* **Out-of-the-Box Streaming Data**: Includes `streaming_generator()` for synthetic regression data—no extra setup required.
* **Hyper-Epochs**: Perform N full curriculum runs (hyper-epochs) with periodic resets.
* **Sinusoidal Curriculum**: Smoothly ramp your curriculum parameter (0→1→0 each hyper-epoch) via a `SinusoidalCurriculum` callback.
* **One-Cycle LR Schedule**: Fast warmup + cosine decay for stable, efficient optimization.
* **Dual Validation**: Tracks both core validation loss (`val_loss`) and curriculum-specific validation (`val_curr`) via a custom callback.
* **Integrated Plotting**: Generates training vs. validation curves at the end of each run using Matplotlib (saved as `training_plot.png`).
* **Minimal**: \~100 lines of code in a single `MicroCurr-V3.py` file under MIT license.

## 🔧 Prerequisites

* Python 3.8+
* [TensorFlow 2.x](https://www.tensorflow.org/)
* [TensorFlow Probability](https://www.tensorflow.org/probability)
* NumPy
* Matplotlib

```bash
pip install tensorflow tensorflow-probability numpy matplotlib
```

## ⚙️ Configuration

At the top of `MicroCurr-V3.py`, tune your experiment:

```python
HYPER_EPOCHS       = 5        # Number of full curriculum cycles
EPOCHS_PER_HYPER   = 20       # Epochs per cycle
BATCH_SIZE         = 128
STEPS_PER_EPOCH    = 200

# Curriculum endpoints
OBLATENESS_END     = 0.5
ROT_FREQ_END       = 2.0
SIGMOID_SCALE      = 0.5

# One-Cycle LR parameters
BASE_LR            = 1e-4
PEAK_LR            = 2.5e-4
MIN_LR             = 1e-6
PCT_WARMUP         = 0.15
```

## ▶️ Running

Clone the repo and run:

```bash
git clone https://github.com/rocketbombs/MicroCurr.git
cd MicroCurr
python MicroCurr-V3.py
```

* Progress logs include `loss`, `val_loss`, and `val_curr` each epoch.
* At completion, you’ll see a summary in the console and a `training_plot.png` in your working directory.

## 📈 Example Results

After 5 hyper-epochs, a typical run prints:

```
loss       floor 0.0283 | tail 0.0401 | drift 0.0117
val_loss   floor 0.0422 | tail 0.0592 | drift 0.0170
val_curr   floor 0.0247 | tail 0.0416 | drift 0.0169
```


![Training Plot](training_plot.png)


## 🔄 Callbacks

* **SinusoidalCurriculum**
  Adjusts curriculum parameters each epoch and resets each hyper-epoch.

* **ValCurrCB**
  Runs mini-validation on current curriculum difficulty and reports a separate Huber loss.

## 🎨 Customization

* Swap in different synthetic data generators or feed in your own dataset.
* Experiment with linear, exponential, or custom curriculum schedules.
* Modify the learning-rate schedule or swap in more complex model architectures.

## 📄 License

MIT License – see [LICENSE](LICENSE) for full text.

---

> Pull requests welcome for new features, improved examples, or performance tweaks!

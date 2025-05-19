# Tiny Curriculum‑Learning Regression – Config v4 (with Hyper‑Epochs)
# -----------------------------------------------------------------------------
# * Run 10 hyper‑epochs of 100 curriculum epochs each
# * Curriculum sigmoid resets at each hyper‑epoch boundary
# * One‑Cycle LR unchanged
# -----------------------------------------------------------------------------

import numpy as np, tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, callbacks, losses

# ----------------------- HYPER-EPOCH CONFIG -------------------------------
HYPER_EPOCHS       = 5        # number of full curriculum runs
EPOCHS_PER_HYPER   = 20       # curriculum epochs per hyper
TOTAL_EPOCHS       = HYPER_EPOCHS * EPOCHS_PER_HYPER

# --------------------------- TRAIN CONFIG ----------------------------------
BATCH_SIZE         = 128
STEPS_PER_EPOCH    = 200
WINDOW             = 5         # moving‑avg window for live plots

# Curriculum parameters -----------------------------------------------------
OBLATENESS_END     = 0.5
ROT_FREQ_END       = 2.0
SIGMOID_SCALE      = 0.5       # shapes schedule within each hyper

# --------------------------- LR SCHEDULE ------------------------------------
BASE_LR            = 1e-4
PEAK_LR            = 2.5e-4
MIN_LR             = 1e-6
PCT_WARMUP         = 0.15
TOTAL_STEPS        = TOTAL_EPOCHS * STEPS_PER_EPOCH
warmup_steps       = int(TOTAL_STEPS * PCT_WARMUP)

a = tf.constant(np.pi, dtype=tf.float32)
class OneCycleSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr):
        super().__init__()
        self.peak_lr, self.total_steps = peak_lr, total_steps
        self.warmup_steps, self.min_lr = warmup_steps, min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm_lr = (self.peak_lr / tf.cast(tf.maximum(1, self.warmup_steps), tf.float32)) * step
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / tf.cast(tf.maximum(1, decay_steps), tf.float32)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cos_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + tf.cos(a * progress))
        return tf.where(step < self.warmup_steps, warm_lr, cos_lr)

LR_SCHED   = OneCycleSchedule(PEAK_LR, TOTAL_STEPS, warmup_steps, MIN_LR)
OPTIMIZER  = tf.keras.optimizers.Adam(learning_rate=LR_SCHED, clipnorm=0.7)

# ----------------------- CALLBACKS -----------------------------------------
# Changed keras.callbacks.Callback to callbacks.Callback
class SinusoidalCurriculum(callbacks.Callback):
    def __init__(self, hyper_epochs, update_fn):
        """
        hyper_epochs: # of epochs per cycle (e.g. 100)
        update_fn:   function(scale: float) that applies your curriculum params
        """
        super().__init__()
        self.hyper_epochs = hyper_epochs
        self.update_fn = update_fn

    def on_epoch_begin(self, epoch, logs=None):
        # map epoch → local progress [0,1] in current hyper-epoch
        local_epoch = epoch % self.hyper_epochs
        p = local_epoch / float(self.hyper_epochs)
        # sinusoidal ramp: 0→1→0 over each cycle
        scale = (1 - np.cos(2 * np.pi * p)) / 2
        # apply to your curriculum parameters (e.g. max_oblateness, freq, sigmoid_scale…)
        self.update_fn(scale)

class ValCurrCB(callbacks.Callback):
    def __init__(self, steps=50):
        super().__init__(); self.steps=steps; self.gen=make_val_curr_gen(); self.m=losses.Huber(delta=0.2)
    def on_epoch_end(self, epoch, logs=None):
        vals = [self.m(y, self.model(x, training=False)).numpy()
                for _ in range(self.steps) for x,y in [next(self.gen)]]
        logs['val_curr'] = float(np.mean(vals))
        print(f"  val_curr: {logs['val_curr']:.4f}")

# ----------------------- MODEL ARCHITECTURE -------------------------------
INPUT_DIM = 9
act = tf.keras.activations.relu
inputs = layers.Input(shape=(INPUT_DIM,))
# residual blocks...
x = layers.Dense(256, activation=act)(inputs)
h = layers.Dense(64, activation=act)(x)
h = layers.Dense(64)(h)
proj = layers.Dense(64, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Activation(act)(h + proj)
h2 = layers.Dense(64, activation=act)(x)
h2 = layers.Dense(64)(h2)
x = layers.Activation(act)(h2 + x)
# tail
x = layers.Dense(16, activation=act)(x)
x = layers.Dense(8,  activation=act)(x)
outputs = layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=OPTIMIZER, loss=tf.keras.losses.Huber(delta=0.2))

# ----------------------- DATA & TRAIN -------------------------------------
train_ds = (tf.data.Dataset.from_generator(curriculum_generator,
            output_types=(tf.float32,tf.float32),
            output_shapes=((None,9),(None,1)))
            .repeat().prefetch(tf.data.AUTOTUNE))
val_core_ds = (tf.data.Dataset.from_generator(val_core_gen,
              output_types=(tf.float32, tf.float32),
              output_shapes=((None, 9), (None, 1)))
              .repeat())

if __name__ == "__main__":
    history = model.fit(
        train_ds,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs          = TOTAL_EPOCHS,
        validation_data = val_core_ds,
        validation_steps= 100,
        callbacks       = [CurriculumCB(), ValCurrCB()],
        verbose         = 1,
    ).history

    # summarize results
    import numpy as np
    def summarize(hist):
        for k in ('loss', 'val_loss', 'val_curr'):
            arr   = np.array(hist[k])
            floor = arr[:15].min()
            tail  = arr[-10:].mean()
            print(f"{k:9s}  floor {floor:.4f} | tail {tail:.4f} | drift {tail-floor:.4f}")
    summarize(history)
    # ----------------------- Plot ---------------------------------------------
import matplotlib.pyplot as plt

loss_arr     = np.array(history['loss'])
val_core_arr = np.array(history['val_loss'])
val_curr_arr = np.array(history['val_curr'])

ma = np.convolve(loss_arr, np.ones(WINDOW) / WINDOW, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(ma, label=f'Train ({WINDOW}-pt MA)')
plt.plot(val_core_arr, label='val_core')
plt.plot(val_curr_arr, label='val_curr')
plt.ylim(0, max(np.concatenate([ma, val_core_arr, val_curr_arr])) * 1.1)
plt.ylabel('Huber Loss')
plt.xlabel(f'Epoch (>{WINDOW - 1} skipped in MA)')
plt.title('Loss Curves – Train vs. Validation (Core & Current)')
plt.legend()
plt.grid(True)
plt.show()
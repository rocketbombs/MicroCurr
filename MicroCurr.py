import collections, random, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from typing import Deque, Tuple
from tensorflow.keras import layers, regularizers, callbacks, optimizers

"""
Hyper‑Epoch Training Script – sigmoid curriculum ramp
----------------------------------------------------
* Five hyper‑epochs, 20 inner epochs each (total 100 epochs)
* Curriculum difficulty now follows a logistic (sigmoid) instead of linear
  – stays flat for the first couple of hypers, then ramps sharply
* All other behaviour unchanged from the previous version
"""

# --------------------------- GLOBAL CONFIG ---------------------------------
NUM_HYPERS        = 5               # complete curriculum passes
EPOCHS_PER_HYPER  = 20              # inner epochs per hyper‑epoch
TOTAL_EPOCHS      = NUM_HYPERS * EPOCHS_PER_HYPER

BATCH_SIZE        = 128
STEPS_PER_EPOCH   = 200             # faster convergence per user request
WINDOW            = 5               # moving‑average window for loss plot

# Curriculum targets ---------------------------------------------------------
OBLATENESS_END    = 0.5
ROT_FREQ_END      = 2.0
HYPER_SIGMOID_K   = 10.0            # >0: larger ⇒ later, steeper ramp

# Optimiser / LR -------------------------------------------------------------
INITIAL_LR        = 1e-3            # cosine starts here
MIN_LR            = 1e-5            # alpha = 0.01
CLIP_NORM         = 1.0

# Replay --------------------------------------------------------------------
REPLAY_FRAC       = 0.55
REPLAY_POOL       = 2000

# ---------------------- Global curriculum state ----------------------------
global_max_oblateness      = 0.0
global_rotation_frequency  = 0.0

# ------------------------ Data generation ----------------------------------

def generate_oblate_sample(t: float,
                           n_positive: int = 64,
                           n_negative: int = 64,
                           r_max: float = 1.0,
                           center: Tuple[float, float] = (0, 0),
                           jitter: float = 0.05):
    """Generate (inputs, signed‑distance) samples for time slice *t*."""
    obl   = global_max_oblateness * (1 - np.cos(t * global_rotation_frequency))
    angle = t * global_rotation_frequency * np.pi

    scale_x, scale_y = 1.0, max(1e-6, 1.0 - obl)
    c, s = np.cos(angle), np.sin(angle)
    R, R_inv = np.array([[c, -s], [s, c]]), np.array([[c, s], [-s, c]])
    S_inv = np.diag([1.0 / scale_x, 1.0 / scale_y])
    base_r = np.abs(np.sin(t)) * r_max

    # Positive points on boundary
    th = np.linspace(0, 2 * np.pi, n_positive, endpoint=False)
    pos = np.stack([base_r * scale_x * np.cos(th), base_r * scale_y * np.sin(th)], 1)
    pos_w = (R @ pos.T).T
    x_pos = center[0] + pos_w[:, 0] + np.random.normal(0, jitter, n_positive)
    y_pos = center[1] + pos_w[:, 1] + np.random.normal(0, jitter, n_positive)

    # Negative samples
    max_ext = 1.5 * r_max * max(scale_x, scale_y)
    r_rand = np.random.uniform(0, max_ext, n_negative)
    th_rand = np.random.uniform(0, 2 * np.pi, n_negative)
    neg = np.stack([r_rand * np.cos(th_rand), r_rand * np.sin(th_rand)], 1)
    neg_w = (R @ neg.T).T
    x_neg = center[0] + neg_w[:, 0] + np.random.normal(0, jitter, n_negative)
    y_neg = center[1] + neg_w[:, 1] + np.random.normal(0, jitter, n_negative)

    x_all = np.concatenate([x_pos, x_neg]); y_all = np.concatenate([y_pos, y_neg])
    t_all = np.full_like(x_all, t)

    # feature engineering
    xy = x_all * y_all; x2, y2 = x_all ** 2, y_all ** 2
    xt, yt = x_all * t_all, y_all * t_all
    sint = np.sin(t) * np.ones_like(x_all)
    inputs = np.stack([x_all, y_all, t_all, xy, x2, y2, xt, yt, sint], 1).astype("float32")

    pts_c = np.stack([x_all - center[0], y_all - center[1]], 1)
    pts_r = (R_inv @ pts_c.T).T
    pts_canon = (S_inv @ pts_r.T).T
    radius = np.linalg.norm(pts_canon, axis=1)
    dist = (radius - base_r).astype("float32")[:, None]
    return inputs, dist

# ------------------------ Generators ---------------------------------------

def curriculum_generator():
    pool: Deque = collections.deque(maxlen=REPLAY_POOL)
    t = 0.0
    while True:
        hard_in, hard_lb = generate_oblate_sample(t, BATCH_SIZE // 2, BATCH_SIZE // 2)
        k = int(REPLAY_FRAC * len(hard_in))
        if len(pool) >= k > 0:
            rep_in, rep_lb = zip(*random.sample(pool, k))
            inputs = np.vstack([hard_in, np.array(rep_in)])
            labels = np.vstack([hard_lb, np.array(rep_lb)])
        else:
            inputs, labels = hard_in, hard_lb
        if global_max_oblateness < OBLATENESS_END or global_rotation_frequency < ROT_FREQ_END:
            pool.extend(list(zip(hard_in, hard_lb)))
        yield inputs, labels
        t += 0.05

# Validation generators -----------------------------------------------------

def val_core_gen():
    t = 5.0
    while True:
        yield generate_oblate_sample(t, BATCH_SIZE // 2, BATCH_SIZE // 2)
        t += 0.05

def make_val_curr_gen(start=5.0):
    t = start
    while True:
        yield generate_oblate_sample(t, BATCH_SIZE // 2, BATCH_SIZE // 2)
        t += 0.05

# ----------------------- MODEL ARCHITECTURE -------------------------------
INPUT_DIM = 9
act = tf.keras.activations.relu
inputs = layers.Input(shape=(INPUT_DIM,))

# Residual stem
x = layers.Dense(256, activation=act)(inputs)

h = layers.Dense(64, activation=act)(x)
h = layers.Dense(64)(h)
proj = layers.Dense(64, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Activation(act)(h + proj)

h2 = layers.Dense(64, activation=act)(x)
h2 = layers.Dense(64)(h2)
x = layers.Activation(act)(h2 + x)

# Tail
x = layers.Dense(16, activation=act)(x)
x = layers.Dense(8,  activation=act)(x)
outputs = layers.Dense(1)(x)

# Optimiser – cosine decay resets each hyper‑epoch --------------------------
COS_DECAY_STEPS = EPOCHS_PER_HYPER * STEPS_PER_EPOCH
lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate = INITIAL_LR,
    decay_steps           = COS_DECAY_STEPS,
    alpha                 = MIN_LR / INITIAL_LR,
)
OPTIMIZER = optimizers.Adam(learning_rate=lr_schedule, clipnorm=CLIP_NORM)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=OPTIMIZER, loss=tf.keras.losses.Huber(delta=0.2))

# ----------------------- DATASETS -----------------------------------------
train_ds = (tf.data.Dataset.from_generator(curriculum_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, 9), (None, 1)))
            .repeat().prefetch(tf.data.AUTOTUNE))

val_core_ds = (tf.data.Dataset.from_generator(val_core_gen,
              output_types=(tf.float32, tf.float32),
              output_shapes=((None, 9), (None, 1)))
              .repeat())

# ----------------------- TRAINING – HYPER‑EPOCH LOOP ----------------------


def summarize(hist):
    for k in ('loss', 'val_loss', 'val_curr'):
        arr   = np.array(hist[k])
        floor = arr[:15].min()
        tail  = arr[-10:].mean()
        print(f"{k:9s}  floor {floor:.4f} | tail {tail:.4f} | drift {tail-floor:.4f}")


def sigmoid_ramp(idx: int, total: int, k: float = HYPER_SIGMOID_K) -> float:
    """Return a curriculum multiplier in [0,1] for hyper‑index *idx* (0‑based)."""
    x = (idx + 0.5) / total                                   # centre of the hyper bin
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))


if __name__ == "__main__":
    all_hist = collections.defaultdict(list)

    for hyper in range(NUM_HYPERS):
        pct = sigmoid_ramp(hyper, NUM_HYPERS)
        globals()['global_max_oblateness']     = OBLATENESS_END * pct
        globals()['global_rotation_frequency'] = ROT_FREQ_END * pct

        print(f"\n=== Hyper‑Epoch {hyper + 1}/{NUM_HYPERS} » pct={pct:.3f} | max_obl={global_max_oblateness:.3f} | rot_freq={global_rotation_frequency:.3f} ===")

        history = model.fit(
            train_ds,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs          = EPOCHS_PER_HYPER,
            validation_data = val_core_ds,
            validation_steps= 100,
            callbacks       = [CurriculumCB(), ValCurrCB()],
            verbose         = 1,
        ).history

        for k, v in history.items():
            all_hist[k].extend(v)

    # ----------------------- SUMMARY & PLOT -------------------------------
    summarize(all_hist)

    loss_arr     = np.array(all_hist['loss'])
    val_core_arr = np.array(all_hist['val_loss'])
    val_curr_arr = np.array(all_hist['val_curr'])

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

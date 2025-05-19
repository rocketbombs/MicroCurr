# Tiny Curriculum‑Learning Regression – Config v4 (with Hyper‑Epochs)
# -----------------------------------------------------------------------------
# * Run 10 hyper‑epochs of 100 curriculum epochs each
# * Curriculum sigmoid resets at each hyper‑epoch boundary
# * One‑Cycle LR unchanged
# -----------------------------------------------------------------------------

import numpy as np, tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, callbacks, losses

# ------------------------ Data generation ----------------------------------

def generate_oblate_sample(t: float,
                           n_positive: int = 64,
                           n_negative: int = 64,
                           r_max: float = 1.0,
                           center: Tuple[float,float] = (0,0),
                           jitter: float = 0.05):
    obl   = global_max_oblateness * (1 - np.cos(t*global_rotation_frequency))
    angle = t * global_rotation_frequency * np.pi

    scale_x, scale_y = 1.0, max(1e-6, 1.0-obl)
    c, s = np.cos(angle), np.sin(angle)
    R, R_inv = np.array([[c,-s],[s,c]]), np.array([[c,s],[-s,c]])
    S_inv = np.diag([1.0/scale_x, 1.0/scale_y])
    base_r = np.abs(np.sin(t))*r_max

    # positive points on boundary
    th = np.linspace(0,2*np.pi,n_positive,endpoint=False)
    pos = np.stack([base_r*scale_x*np.cos(th), base_r*scale_y*np.sin(th)],1)
    pos_w = (R@pos.T).T
    x_pos = center[0]+pos_w[:,0]+np.random.normal(0,jitter,n_positive)
    y_pos = center[1]+pos_w[:,1]+np.random.normal(0,jitter,n_positive)

    # negatives
    max_ext = 1.5*r_max*max(scale_x,scale_y)
    r_rand  = np.random.uniform(0,max_ext,n_negative)
    th_rand = np.random.uniform(0,2*np.pi,n_negative)
    neg     = np.stack([r_rand*np.cos(th_rand), r_rand*np.sin(th_rand)],1)
    neg_w   = (R@neg.T).T
    x_neg = center[0]+neg_w[:,0]+np.random.normal(0,jitter,n_negative)
    y_neg = center[1]+neg_w[:,1]+np.random.normal(0,jitter,n_negative)

    x_all = np.concatenate([x_pos,x_neg]); y_all = np.concatenate([y_pos,y_neg])
    t_all = np.full_like(x_all,t)

    xy  = x_all*y_all; x2, y2 = x_all**2, y_all**2
    xt, yt = x_all*t_all, y_all*t_all
    sint = np.sin(t)*np.ones_like(x_all)
    inputs = np.stack([x_all,y_all,t_all,xy,x2,y2,xt,yt,sint],1).astype('float32')

    pts_c = np.stack([x_all-center[0], y_all-center[1]],1)
    pts_r = (R_inv@pts_c.T).T
    pts_canon = (S_inv@pts_r.T).T
    radius = np.linalg.norm(pts_canon,axis=1)
    dist = (radius-base_r).astype('float32')[:,None]
    return inputs, dist

# ------------------------ Generators ---------------------------------------

def curriculum_generator():
    pool: Deque = collections.deque(maxlen=REPLAY_POOL)
    t = 0.0
    while True:
        hard_in, hard_lb = generate_oblate_sample(t,BATCH_SIZE//2,BATCH_SIZE//2)
        k = int(REPLAY_FRAC*len(hard_in))
        if len(pool)>=k>0:
            rep_in, rep_lb = zip(*random.sample(pool,k))
            inputs = np.vstack([hard_in,np.array(rep_in)])
            labels = np.vstack([hard_lb,np.array(rep_lb)])
        else:
            inputs, labels = hard_in, hard_lb
        if global_max_oblateness<OBLATENESS_END or global_rotation_frequency<ROT_FREQ_END:
            pool.extend(list(zip(hard_in,hard_lb)))
        yield inputs, labels
        t += 0.05

# Validation generators

def val_core_gen():
    t = 5.0
    while True:
        yield generate_oblate_sample(t,BATCH_SIZE//2,BATCH_SIZE//2)
        t += 0.05

def make_val_curr_gen(start=5.0):
    t = start
    while True:
        yield generate_oblate_sample(t,BATCH_SIZE//2,BATCH_SIZE//2)
        t += 0.05
        
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

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import neurite as ne
from tqdm import tqdm
from scipy.ndimage import zoom
import voxelmorph as vxm 
import generators
import losses
import render_output

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def plot_history(hist):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history['loss'], '.-', label='Training Loss')
    plt.plot(hist.epoch, hist.history['val_loss'], '.-', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('VoxelMorph Training Loss')
    
    plt.savefig(output_dir + prefix + '_loss.png')

# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

file_path = '/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy'  # input image directory
output_dir = '/home/sarahl/Documents/Fall Rotation/VoxelMorph/out/'                           # output model directory

prefix = 'Masked'                                 # output model name prefix
gpus = [0]
device = 'cuda:0'
cudnn_nondet = True                             # disable cudnn determinism - might slow down training
bidirectional = False                           # enable bidirectional cost function (not implemented)
batch_size = 1
lr = 1e-5                                       # learning rate (default: 1e-4)
epochs = 50                                     # number of training epochs (default: 1500)
steps_per_epoch = 150                           # number of training batches per epoch (default: 100)
val_steps_per_epoch = 30
initial_epoch = 0                               # initial epoch number (default: 0)
debug = False                                   # when debug = True, script only loads two scans and trains for two epochs
ncc = False
masked = True


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
files = os.listdir(file_path)
img_files = [file_path + '/' + f for f in files if f.endswith('.npy') and len(f) == 19] # shape (T, Z, Y, X)
mask_files = [file_path + '/' + f for f in files if f.endswith('biv.npy') and len(f) == 23]

#Interpolation parameters: input image dimensions (px x px)
ht=128 #512 
wd=128 #512
dp=128
fixed = []
moving = []

# load images from paths and arrange into ordered 'fixed' and 'moving' lists
with tqdm(total=len(img_files)) as pbar:
    for i, img_path in enumerate(img_files):
        print("Loading 3D US file: " + img_path)
        scan = np.load(img_path, allow_pickle=True)
        if masked: 
            mask_path = mask_files[i]
            print("Load 3D mask: " + mask_path)
            mask = np.load(mask_path, allow_pickle=True)

        num_frames = scan.shape[0]
        if debug: num_frames = 25

        with tqdm(total=num_frames) as pbar2:
            for frame_num in range(num_frames):
                fr = scan[frame_num,:,:,:]
                factors = [128/s for s in fr.shape]
                frame = zoom(fr, factors, order=1)
                if masked:
                    msk_fr = mask[frame_num, :, :]
                    msk_frame = zoom(msk_fr, factors, order=1)
                    frame = frame * msk_frame
                
                if frame_num > 0:
                    fixed.append(frame / np.max(np.absolute(frame)))
                if frame_num < num_frames-1:
                    moving.append(frame / np.max(np.absolute(frame)))
                pbar2.update()
        pbar2.close()

        print("Total Loading Progress: ")
        pbar.update()
        # if debug == True: break
pbar.close()
print()

fixed = np.array(fixed) 
moving = np.array(moving) 

train_fixed, val_fixed, train_moving, val_moving = train_test_split(moving, fixed, test_size=0.3, random_state=50)

# prints the number of image pairs for training and validation sets
print("Training Dataset Length: %d" % len(train_fixed))
print("Validation Dataset Length: %d" % len(val_fixed))

train_generator = generators.vol_generator(train_moving, train_fixed, batch_size=batch_size)
val_generator = generators.vol_generator(val_moving, val_fixed, batch_size=batch_size)

# UNCOMMENT TO VISUALIZE LOADED DATA
# while True:
#     input, _ = next(train_generator)
#     plt.imshow(input[0][0,:,64,:], cmap="gray", aspect="auto", origin="lower")
#     plt.colorbar(label="Intensity")
#     plt.savefig(output_dir + '/test_slice_xz.png')
#     plt.close()
    
#     plt.imshow(input[0][0,:,:,64], cmap="gray", aspect="auto", origin="lower")
#     plt.colorbar(label="Intensity")
#     plt.savefig(output_dir + '/test_slice_xy.png')
#     plt.close()

#     plt.imshow(input[0][0,64,:,:], cmap="gray", aspect="auto", origin="lower")
#     plt.colorbar(label="Intensity")
#     plt.savefig(output_dir + '/test_slice_yz.png')
#     plt.close()

#     exit()

# ----------------------- MODEL CREATION -----------------------

# configure unet features 
nb_features = [
    [16, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 32, 16]  # decoder features
]

# build model using custom model Vxm4D
inshape = next(train_generator)[0][0].shape[1:-1]
vxm_model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=nb_features,
    bidir=bidirectional,
    int_steps=7, # number of integration steps (default: 7)
) #bmode_rf_network.Vxm4D(inshape, nb_features, int_steps=0)

# instantiate losses
if ncc:
    loss_weights = [-1, 0.01]   
    losses = [vxm.losses.NCC(win=[10, 45]).loss, vxm.losses.Grad('l2').loss]
else:
    loss_weights = [100, 5]
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
vxm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=losses,
    loss_weights=loss_weights,
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
    factor=0.1,
    patience=3,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.0
)

early_stop = EarlyStopping(monitor='val_loss',
    min_delta=0.00001,
    patience=15,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

# ----------------------- GPU CHECKUP + WARMUP -----------------------

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# small warm-up and test matmul
a = tf.random.normal([1024, 1024])
for _ in range(5):
    _ = tf.matmul(a, a)

# simple warm-up
for _ in range(5):
    _ = vxm_model((tf.zeros([1, *inshape, 1]), tf.zeros([1, *inshape, 1])))

# ----------------------- MODEL TRAINING -----------------------

# val_steps_per_epoch = len(moving_bmode // batch_size)
# steps_per_epoch = 5*val_steps_per_epoch                # number of training batches per epoch (default: 100)

if debug == True: 
    epochs = 5
    steps_per_epoch = 2
    val_steps_per_epoch = 1

# vxm_model.summary()

hist = vxm_model.fit(train_generator, 
                     epochs=epochs, 
                     steps_per_epoch=steps_per_epoch, 
                     verbose=1,
                     validation_data=val_generator,
                     validation_steps=val_steps_per_epoch,
                     callbacks=[reduce_lr, early_stop]) #, tqdm_progress])
    
vxm_model.save_weights(output_dir + prefix + ".weights.h5")

plot_history(hist)

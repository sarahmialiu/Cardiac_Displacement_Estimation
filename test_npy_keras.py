import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import neurite as ne
from tqdm import tqdm
import seaborn as sns
import glob
from scipy.io import loadmat
import pandas as pd
import mat73

# import voxelmorph with pytorch backend
# os.environ['NEURITE_BACKEND'] = 'pytorch'
# os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import bmode_rf_network
import generators
import losses


# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

folders_path = 'Jad_RFData'                     # input image directory
weights_path = 'Bmode_RF_MSE.weights.h5'
prefix = 'Bmode_RF_MSE'
gpus = [0]
device = 'cuda:0'
cudnn_nondet = True                             # disable cudnn determinism - might slow down training
bidirectional = False                           # enable bidirectional cost function
batch_size = 1
ncc = False

# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
wd=176
ht=1312
skip = False
fixed_bmode = []
moving_bmode = []
fixed_rf = []
moving_rf = []

folders_paths = glob.glob(folders_path + '/*')

vert_displacements = []
hzn_displacements = []
filenames = []

for sim_num in range(1):
    folder_path = folders_paths[random.randint(0, len(folders_paths)-1)]

    sims_paths = glob.glob(folder_path + '/*')
    sim_path = sims_paths[random.randint(0, len(sims_paths)-1)]
    # sim_path = 'Jad_RFData\\\Random Parameters\\10thick_8deep'

    bmode_path = Path(glob.glob(sim_path+'/ppb.mat')[0])
    bmode = loadmat(bmode_path)['BmodeRFScanConv']
    num_slices = bmode.shape[2]

    mask_path = Path(glob.glob(sim_path+'/processed_data.mat')[0])
    mask_files = loadmat(mask_path)['maskMat']
    mask = mask_files[...,0]

    rf_path = Path(glob.glob(sim_path+'/ConvRF.mat')[0])
    rf = loadmat(rf_path)['rfScanConv']
    num_rf_slices = rf.shape[2]
    if num_rf_slices < num_slices: skip = True


    for slice_num in range(num_slices):
        slice = bmode[:,:,slice_num]
        slice = np.nan_to_num(slice, nan=np.nanmax(slice)) # changing nans to lowest value
        slice = cv2.resize(slice, (wd,ht), interpolation=cv2.INTER_NEAREST)
        slice = slice / np.max(np.absolute(slice))

        rf_slice = rf[:,:,slice_num]
        rf_slice = np.nan_to_num(rf_slice, nan=np.nanmax(rf_slice)) # changing nans to lowest value # changing nans to 0  

        rf_slice = np.log1p(np.abs(rf_slice-0.5))
        rf_slice = cv2.resize(rf_slice, (wd,ht), interpolation=cv2.INTER_NEAREST)
        rf_slice = rf_slice / np.max(np.absolute(rf_slice))
        rf_slice = -1 + (rf_slice + 1)/2

        # This block loads consecutive pairs of images:
        if slice_num > 0: 
            fixed_bmode.append(slice)
            fixed_rf.append(rf_slice)
        if slice_num < num_slices-1:
            moving_bmode.append(slice)
            moving_rf.append(rf_slice)      
        
        curr_mask = mask_files[:,:,slice_num-1]
        mask = np.logical_or(mask, curr_mask)

        if slice_num == 0: first_frame = slice

fixed_bmode = np.array(fixed_bmode)
moving_bmode = np.array(moving_bmode)
fixed_rf = np.array(fixed_rf)
moving_rf = np.array(moving_rf)

fixed = np.stack((fixed_bmode, fixed_rf), axis=3) # (slices, wd, ht, 2) <- bmode = 0, rf = 1
moving = np.stack((moving_bmode, moving_rf), axis=3) # (slices, wd, ht, 2)

mask = np.logical_not(mask).astype(int)
mask = cv2.resize(mask, (wd,ht), interpolation=cv2.INTER_NEAREST)

# prints the number of image pairs for training and validation sets
print("Testing Dataset Length: %d" % len(fixed))

test_generator = generators.generator_4D(moving[..., 0], fixed[..., 0], moving[..., 1], fixed[..., 1], batch_size=batch_size)

# ----------------------- MODEL LOADING AND PREDICTION -----------------------

# configure unet features 
nb_features = [
    [16, 32, 32, 32],               # encoder features
    [32, 32, 32, 32, 32, 32, 16]    # decoder features
]

# build model using VxmDense
inshape = moving_bmode.shape[1:]
vxm_model = bmode_rf_network.Vxm4D(inshape, nb_features, int_steps=0)

# instantiate losses
if ncc:
    loss_weights = [-1, 0.01]   
    losses = [vxm.losses.NCC(win=[10, 45]).loss, vxm.losses.Grad('l2').loss]
else:
    loss_weights = [100, 5]
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

vxm_model.load_weights(weights_path)

total_flow = np.zeros((ht, wd, 2))

# testing loop; iterates through entire test generator
for i in range(len(moving)):
    val_input, _ = next(test_generator)

    val_pred = vxm_model.predict(val_input, verbose=0)
    pred_flow = val_pred[1].squeeze()

    pred_flow[...,0] = 0.2058 * pred_flow[...,0] # scale factors for converting from pixels to mm
    pred_flow[...,1] = 1 * pred_flow[...,1]

    total_flow = total_flow + pred_flow # add predicted flow for each image pair to overall cumulative flow

    nonzeros = first_frame != 0.00
    nonzero_mask = np.logical_not(mask).astype(int) * nonzeros
    
    vert_nonzeros = total_flow[...,0][nonzero_mask != 0.000]
    hzn_nonzeros = total_flow[...,1][nonzero_mask != 0.000]

    vert_displacements.append(np.mean(np.absolute(vert_nonzeros)))
    hzn_displacements.append(np.mean(np.absolute(hzn_nonzeros)))
    filenames.append(sim_path)

df = pd.DataFrame({'Simulation': filenames, 'Vertical': vert_displacements, 'Horizontal': hzn_displacements})
# df.to_csv(Path("out\\displacements_sim_NCC.csv"), index=False)

# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

# Moving/Fixed/Moved
images = [cv2.resize(img[0, :, :, 0], (512, 512), interpolation=cv2.INTER_NEAREST) for img in val_input + tuple(val_pred)] 
titles = ['moving', 'fixed', 'moving rf', 'fixed rf', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
plt.savefig(prefix + '.png')

# Cumulative Flow
# ne.plot.flow([val_pred[1].squeeze()[::4, ::4, :]], width=5) # one frame
flw = np.zeros([256, 256, 2])
flw[..., 0] = cv2.resize(total_flow[..., 0], (256, 256), interpolation=cv2.INTER_NEAREST)
flw[..., 1] = cv2.resize(total_flow[..., 1], (256, 256), interpolation=cv2.INTER_NEAREST)
ne.plot.flow([5*flw[::4, ::4, :] / np.max(np.absolute(total_flow))], width=10)
plt.savefig(prefix + '_flow.png')

# Displacement Heatmaps
# fig, ax = plt.subplots(1, 3)
# fig.suptitle(bmode_path)

# sns.heatmap(pred_flow[::4,::4,0], ax=ax[0], annot=False, cmap="viridis")
# ax[0].set_title("Relative Displacement (Vertical)")
# ax[0].axis('off')

# sns.heatmap(pred_flow[::4,::4,1], ax=ax[1], annot=False, cmap="viridis")
# ax[1].set_title("Relative Displacement (Horizontal)")
# ax[1].axis('off')

# ax[2].imshow(val_pred[0][0,:,:,0], cmap='gray', aspect='auto')
# ax[2].set_title("Sample Predicted Image")
# ax[2].axis('off')

# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(1, 3)
# fig.suptitle(bmode_path)

# sns.heatmap(total_flow[::2,::2,0], ax=ax[0], annot=False, cmap="viridis")
# ax[0].set_title("Relative Cumulative Displacement (Vertical)")
# ax[0].axis('off')

# sns.heatmap(total_flow[::2,::2,1], ax=ax[1], annot=False, cmap="viridis")
# ax[1].set_title("Relative Cumulative Displacement (Horizontal)")
# ax[1].axis('off')

# ax[2].imshow(val_pred[0][0,:,:,0], cmap='gray', aspect='auto')
# ax[2].set_title("Sample Predicted Image")
# ax[2].axis('off')

# plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(1,3, figsize=(18,5))
fig.suptitle(sim_path)

ax[0].imshow(np.flipud(first_frame), extent=[0, wd, 0, ht], aspect='auto', cmap='gray')
vert_heat = sns.heatmap(total_flow[...,0], 
                            mask = mask, 
                            #vmin=-10, vmax=15, 
                            center=0, 
                            ax=ax[0], 
                            annot=False, 
                            cmap="vlag", 
                            #alpha=0.6, 
                            zorder=2)
vert_bar = vert_heat.collections[0].colorbar
vert_bar.ax.tick_params(labelsize=20)
vert_bar.set_label('Displacement (pixels)', size=20)
ax[0].axis('off')
ax[0].set_title('Vertical', fontsize=20)

ax[1].imshow(np.flipud(first_frame), extent=[0, wd, 0, ht], aspect='auto', cmap='gray')
hzn_heat = sns.heatmap(total_flow[...,1], 
                            mask = mask, 
                            #vmin=-10, vmax=15, 
                            center=0, 
                            ax=ax[1], 
                            annot=False, 
                            cmap="vlag", 
                            #alpha=0.6, 
                            zorder=2)
hzn_bar = hzn_heat.collections[0].colorbar
hzn_bar.ax.tick_params(labelsize=20)
hzn_bar.set_label('Displacement (pixels)', size=20)
ax[1].axis('off')
ax[1].set_title('Horizontal', fontsize=20)

ax[2].imshow(first_frame, cmap='gray', aspect='auto')
ax[2].set_title("B-Mode", fontsize=20)
ax[2].axis('off')

plt.tight_layout()
plt.savefig(prefix + '_heatmap.png')

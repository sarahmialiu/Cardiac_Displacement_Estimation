import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import neurite as ne
from scipy.io import loadmat
import glob
import seaborn as sns

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators

# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

folders_path = 'Jad_RFData'                               # input images directory
weights_path = 'out\JadSim_Outputs\model_weights_NCC.pth' # model weights file path
bidirectional = False                                     # enable bidirectional cost function
batch_size = 1
gpus = [0]
device = 'cuda:0'


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
wd= 176 
ht= 1312 
test_fixed = []
test_moving = []

print("Loading B mode Data")
folders = glob.glob(folders_path + '\*')
folder_num = np.random.randint(0, len(folders))

sims = glob.glob(folders[folder_num] + '\*')
sim_num = np.random.randint(0, len(sims))

bmode_path = Path(glob.glob(sims[sim_num]+'\ppb.mat')[0])
bmode = loadmat(bmode_path)['BmodeRFScanConv']
num_slices = bmode.shape[2]
            
for slice_num in range(bmode.shape[2]):
    slice = bmode[:,:,slice_num]
    slice=cv2.resize(slice, (wd,ht), interpolation=cv2.INTER_NEAREST)     
    # slice = slice[455:1175, 32:144]           
    slice = np.nan_to_num(slice, nan=0.0) 

    if slice_num > 0: 
        test_fixed.append(slice / np.max(np.absolute(slice)))
    if slice_num < num_slices-1: 
        test_moving.append(slice / np.max(np.absolute(slice)))


test_fixed = np.array(test_fixed) 
test_moving = np.array(test_moving) 

print("Training Dataset Length: %d" % len(test_fixed))
print()

test_generator = generators.custom_generator(test_moving, test_fixed, batch_size=batch_size)


# ----------------------- MODEL LOADING AND PREDICTION -----------------------

inshape = next(test_generator)[0][0].shape[1:-1]

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# enc_nf = [16, 16, 32, 64] 
# dec_nf = [64, 64, 64, 32, 16, 16, 16] 

model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=False,
        int_steps=7, # number of integration steps (default: 7)
        int_downsize=2 # flow downsample factor for integration (default: 2)
    )

model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()
total_flow = np.zeros((656, 88, 2)) # wd

for i in range(len(test_fixed)):
    test_input, _ = next(test_generator)

    inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in test_input]

    # run inputs through the model to produce a warped image and flow field
    outputs = model(*inputs) # tuple
    pred_img = outputs[0].detach().cpu().numpy()
    pred_flow = outputs[1].permute(0, 2, 3, 1).detach().cpu().numpy()

    total_flow = total_flow + pred_flow[0,...]


# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

# Moving/Fixed/Moved
images = [cv2.resize(test_input[0][0,:,:,0], (500,500), cv2.INTER_NEAREST),
            cv2.resize(test_input[1][0,:,:,0], (500,500), cv2.INTER_NEAREST),
            cv2.resize(pred_img[0,0,:,:], (500,500), cv2.INTER_NEAREST),
            cv2.resize(pred_flow[0,:,:,0], (500,500), cv2.INTER_NEAREST)]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# Cumulative Flow
ne.plot.flow([total_flow[::8,:,:]], width=5) # flow[..., 0] = flow in y-direction, flow[..., 1] = flow in x-direction

# Displacement Heatmaps
fig, ax = plt.subplots(1, 3)
fig.suptitle(sims[sim_num])

sns.heatmap(pred_flow[0,::4,::4,0], ax=ax[0], annot=False, cmap="viridis")
ax[0].set_title("Cardiac Displacement: dim 0 (Vertical)")

sns.heatmap(pred_flow[0,::4,::4,1], ax=ax[1], annot=False, cmap="viridis")
ax[1].set_title("Cardiac Displacement: dim 1 (Horizontal)")

ax[2].imshow(pred_img[0,0,:,:], cmap='gray', aspect='auto')
ax[2].set_title("Predicted Image at end of Systole")
ax[2].axis('off')

plt.tight_layout()
plt.show()

import os
import numpy as np
import torch
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
#import neurite as ne
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage import zoom
import random

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators 


# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

imgs_path = 'DataVisualization/data/ultrasound 4D npy'                 # input images directory
weights_path = "VoxelMorph/out/model_weights.pth"  # model weights file path
bidirectional = False                                     # enable bidirectional cost function
batch_size = 1
gpus = [0]
device = 'cuda:0'
output_dir = 'VoxelMorph/out'


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare testing data
files = os.listdir(imgs_path)
npy_files = [imgs_path + '/' + f for f in files if f.endswith('.npy') and len(f) == 19]

#Interpolation parameters: input image dimensions (px x px)
ht=128 #512 
wd=128 #512
dp=128
fixed = []
moving = []

for i, file_path in enumerate(npy_files):
    scan = np.load(file_path, allow_pickle=True)
    num_frames = scan.shape[0]

    with tqdm(total=num_frames) as pbar:
        for frame_num in range(num_frames):
            fr = scan[frame_num,:,:,:]
            factors = [128/s for s in fr.shape]
            frame = zoom(fr, factors, order=1)
            
            if frame_num < num_frames-1:
                moving.append(frame / np.max(np.absolute(frame)))
            if frame_num > 0:
                fixed.append(frame / np.max(np.absolute(frame)))

            pbar.update()
    pbar.close()
    break # this makes us take only the first scan

test_fixed = np.array(fixed)
test_moving = np.array(moving)

indices = random.sample(range(336), 20)
test_fixed = test_fixed[indices]
test_moving = test_moving[indices]

print("Testing Dataset Length: %d" % len(test_fixed))

test_generator = generators.custom_generator(test_moving, test_fixed, batch_size=batch_size)



# UNCOMMENT TO VISUALIZE TEST DATA
# while True:
#     input, _ = next(test_generator)
#     # MOVING
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

#     # FIXED
#     plt.imshow(input[1][0,:,64,:], cmap="gray", aspect="auto", origin="lower")
#     plt.colorbar(label="Intensity")
#     plt.savefig(output_dir + '/test_slice_xz.png')
#     plt.close()
    
#     plt.imshow(input[1][0,:,:,64], cmap="gray", aspect="auto", origin="lower")
#     plt.colorbar(label="Intensity")
#     plt.savefig(output_dir + '/test_slice_xy.png')
#     plt.close()

#     plt.imshow(input[1][0,64,:,:], cmap="gray", aspect="auto", origin="lower")
#     plt.colorbar(label="Intensity")
#     plt.savefig(output_dir + '/test_slice_yz.png')
#     plt.close()

    # images = [input[0][0,:,:,0], input[1][0,:,:,0]] 
    # titles = ['fixed', 'moving']
    # ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# ----------------------- MODEL LOADING AND PREDICTION -----------------------

inshape = next(test_generator)[0][0].shape[1:-1]

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

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
total_flow = np.zeros((wd//2, wd//2, wd//2, 2))

# testing loop; iterates through entire test generator
for i in range(len(test_fixed)):
    test_input, _ = next(test_generator)

    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in test_input]

    outputs = model(*inputs)
    pred_img = outputs[0].detach().cpu().numpy()
    pred_flow = outputs[1].permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    total_flow = total_flow + pred_flow[0,...] # add predicted flow for each image pair to overall cumulative flow


# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

# Moving/Fixed/Moved
images = [test_input[0][0,:,:,0], test_input[1][0,:,:,0], pred_img[0,0,:,:], pred_flow[0,:,:,0]]
titles = ['moving', 'fixed', 'moved', 'flow']
# ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# Cumulative Flow
# ne.plot.flow([100*total_flow], width=5) # flow[..., 0] = flow in y-direction, flow[..., 1] = flow in x-direction

# Displacement Heatmaps
fig, ax = plt.subplots(1, 3)

sns.heatmap(pred_flow[0,::4,::4,0], ax=ax[0], annot=False, cmap="viridis")
ax[0].set_title("Cardiac Displacement: dim 0 (Vertical)")
ax[0].axis('off')

sns.heatmap(pred_flow[0,::4,::4,1], ax=ax[1], annot=False, cmap="viridis")
ax[1].set_title("Cardiac Displacement: dim 1 (Horizontal)")
ax[1].axis('off')

ax[2].imshow(pred_img[0,0,:,:], cmap='gray', aspect='auto')
ax[2].set_title("Predicted Image at end of Systole")
ax[2].axis('off')

plt.tight_layout()
plt.show()

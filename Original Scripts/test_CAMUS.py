import os
import numpy as np
import torch
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import neurite as ne
import seaborn as sns

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators 


# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

imgs_path = 'CAMUS\img_test_2CH_contrast'                 # input images directory
weights_path = "out\CAMUS_Outputs\model_weights_NCC.pth"  # model weights file path
bidirectional = False                                     # enable bidirectional cost function
batch_size = 1
gpus = [0]
device = 'cuda:0'


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare testing data
us_files = [imgs_path + '\\' + f for f in os.listdir(imgs_path)]

#Interpolation parameters: input image dimensions (px x px)
ht=512 
wd=512
fixed = []
moving = []

for i, file_path in enumerate(us_files):
    scan = tiff.imread(file_path)
    num_slices = scan.shape[0]

    for slice_num in range(num_slices):
        slice = scan[slice_num,:,:]
        slice=cv2.resize(slice, # resize frame by maintaining consistent aspect ratio for each image
                (wd,ht),
                interpolation=cv2.INTER_NEAREST)
        if slice_num > 0:
            fixed.append(slice / np.max(np.absolute(slice)))
        if slice_num < num_slices-1:
            moving.append(slice / np.max(np.absolute(slice)))
    break # this makes us take only the first scan

test_fixed = np.array(fixed)
test_moving = np.array(moving)

print("Testing Dataset Length: %d" % len(test_fixed))

test_generator = generators.custom_generator(test_moving, test_fixed, batch_size=batch_size)


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
total_flow = np.zeros((wd//2, wd//2, 2))

# testing loop; iterates through entire test generator
for i in range(len(test_fixed)):
    test_input, _ = next(test_generator)

    inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in test_input]

    outputs = model(*inputs)
    pred_img = outputs[0].detach().cpu().numpy()
    pred_flow = outputs[1].permute(0, 2, 3, 1).detach().cpu().numpy()

    total_flow = total_flow + pred_flow[0,...] # add predicted flow for each image pair to overall cumulative flow


# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

# Moving/Fixed/Moved
images = [test_input[0][0,:,:,0], test_input[1][0,:,:,0], pred_img[0,0,:,:], pred_flow[0,:,:,0]]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# Cumulative Flow
ne.plot.flow([100*total_flow], width=5) # flow[..., 0] = flow in y-direction, flow[..., 1] = flow in x-direction

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

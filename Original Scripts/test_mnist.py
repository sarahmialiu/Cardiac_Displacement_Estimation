import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import neurite as ne
import seaborn as sns
from sklearn.datasets import fetch_openml

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators

# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

weights_path = 'out\MNIST_Outputs\model_weights_NCC.pth' # model weights file path
bidirectional = False                                     # enable bidirectional cost function
batch_size = 1
gpus = [0]
device = 'cuda:0'

# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare training data
wd=28
ht=28

mnist = fetch_openml('mnist_784', version=1)

mnist_samples = mnist['data'][mnist['target'].astype(int) == 5]
mnist_samples = mnist_samples.astype('float') / 255.0
mnist_samples = np.array(mnist_samples)

fixed = []
moving = []

for i in range(2):
    idx1 = np.random.randint(0, mnist_samples.shape[0])
    fix = mnist_samples[idx1, :].reshape((ht,wd))
    fix = cv2.resize(fix, (ht*4, wd*4), interpolation=cv2.INTER_NEAREST)
    fixed.append(fix)
    
    idx2 = np.random.randint(0, mnist_samples.shape[0])
    mov = mnist_samples[idx2, :].reshape((ht,wd))
    mov = cv2.resize(mov, (ht*4, wd*4), interpolation=cv2.INTER_NEAREST)
    moving.append(mov)          

test_fixed = np.array(fixed) 
test_moving = np.array(moving) 

print("Training Dataset Length: %d" % len(test_fixed))
print()

test_generator = generators.custom_generator(test_fixed, test_moving, batch_size=batch_size)


# ----------------------- MODEL LOADING AND PREDICTION -----------------------

inshape = next(test_generator)[0][0].shape[1:-1]

# enc_nf = [16, 32, 32, 32]
# dec_nf = [32, 32, 32, 32, 32, 16, 16]

enc_nf = [16, 16, 32, 64] 
dec_nf = [64, 64, 64, 32, 16, 16, 16] 

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

test_input, _ = next(test_generator)
inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in test_input]

outputs = model(*inputs, registration=True) # tuple
pred_img = outputs[0].detach().cpu().numpy()
pred_flow = outputs[1].permute(0, 2, 3, 1).detach().cpu().numpy()


# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

# Moving/Fixed/Moved
images = [test_input[0][0,:,:,0], test_input[1][0,:,:,0], pred_img[0,0,:,:], pred_flow[0,:,:,0]]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# Cumulative Flow
registration_outputs = model(*inputs, registration=True)
registration_flow = registration_outputs[1].permute(0, 2, 3, 1).detach().cpu().numpy()

ne.plot.flow(10*registration_flow, width=5) # flow[..., 0] = flow in y-direction, flow[..., 1] = flow in x-direction

# Displacement Heatmaps
print("Generating heatmaps...")
fig, ax = plt.subplots(1, 4)

sns.heatmap(pred_flow[0,::2,::2,0], ax=ax[0], annot=False, cmap="viridis")
ax[0].set_title("Displacement (Vertical)")
ax[0].axis('off')

sns.heatmap(pred_flow[0,::2,::2,1], ax=ax[1], annot=False, cmap="viridis")
ax[1].set_title("Displacement (Horizontal)")
ax[1].axis('off')

ax[2].imshow(test_input[1][0,:,:,0], cmap='gray', aspect='auto')
ax[2].set_title("Fixed Image")
ax[2].axis('off')

ax[3].imshow(test_input[0][0,:,:,0], cmap='gray', aspect='auto')
ax[3].set_title("Moving Image")
ax[3].axis('off')

plt.tight_layout()
plt.show()

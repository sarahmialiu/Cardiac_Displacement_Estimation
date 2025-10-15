import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
import voxelmorph as vxm  # nopep8
import generators
import losses
from render_output import render_output

# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

img_path = '/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30.npy'  # input image directory
output_dir = '/home/sarahl/Documents/Fall Rotation/VoxelMorph/out'                           # output model directory

weights_path = '/home/sarahl/Documents/Fall Rotation/VoxelMorph/out/TEST.weights.h5'
prefix = 'TEST'
gpus = [0]
device = 'cuda:0'
cudnn_nondet = True                             # disable cudnn determinism - might slow down training
bidirectional = False                           # enable bidirectional cost function
batch_size = 1
ncc = False

# ----------------------- DATA PREPROCESSING -----------------------

#Interpolation parameters: input image dimensions (px x px)
ht=128 #512 
wd=128 #512
dp=128
fixed = []
moving = []

# load images from paths and arrange into ordered 'fixed' and 'moving' lists
print("Loading 3D US file: " + img_path)
scan = np.load(img_path, allow_pickle=True)

num_frames = 20 #scan.shape[0]

with tqdm(total=num_frames) as pbar2:
    for frame_num in range(num_frames):
        fr = scan[frame_num,:,:,:]
        factors = [128/s for s in fr.shape]
        frame = zoom(fr, factors, order=1)
        
        if frame_num > 0:
            fixed.append(frame / np.max(np.absolute(frame)))
        if frame_num < num_frames-1:
            moving.append(frame / np.max(np.absolute(frame)))
        pbar2.update()
pbar2.close()

print()

test_fixed = np.array(fixed) 
test_moving = np.array(moving) 

# prints the number of image pairs for training and validation sets
print("Testing Dataset Length: %d" % len(test_fixed))

test_generator = generators.vol_generator(test_moving, test_fixed, batch_size=batch_size)

# ----------------------- MODEL LOADING AND PREDICTION -----------------------

# configure unet features 
nb_features = [
    [16, 32, 32, 32],               # encoder features
    [32, 32, 32, 32, 32, 32, 16]    # decoder features
]

# build model using VxmDense
inshape = test_moving.shape[1:]

vxm_model = vxm.networks.VxmDense(inshape, nb_features, bidir=bidirectional, int_steps=7)

# instantiate losses
if ncc:
    loss_weights = [-1, 0.01]   
    losses = [vxm.losses.NCC(win=[10, 45]).loss, vxm.losses.Grad('l2').loss]
else:
    loss_weights = [100, 5]
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

vxm_model.load_weights(weights_path)

input = np.zeros([num_frames, ht, wd, dp])
pred = np.zeros([num_frames, ht, wd, dp])
hzn_flow = np.zeros([num_frames, ht//2, wd//2, dp//2])
vert_flow = np.zeros([num_frames, ht//2, wd//2, dp//2])

for i in range(len(moving)):
    test_input, _ = next(test_generator)
    input[i] = test_input[1].squeeze() # start visualization at t = 1

    test_pred, test_flow = vxm_model.predict(test_input, verbose=0)
    pred[i] = test_pred.squeeze()
    hzn_flow[i] = test_flow.squeeze()[..., 0]
    vert_flow[i] = test_flow.squeeze()[..., 1]
    # pred_flow = test_pred[1].squeeze()
    
print(input.shape, pred.shape, hzn_flow.shape, vert_flow.shape)

# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

render_output(input, pred, hzn_flow, vert_flow)

# # Moving/Fixed/Moved
# images = [cv2.resize(img[0, :, :, 0], (512, 512), interpolation=cv2.INTER_NEAREST) for img in test_input + tuple(test_pred)] 
# titles = ['moving', 'fixed', 'moving rf', 'fixed rf', 'moved', 'flow']
# ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
# plt.savefig(prefix + '.png')

# # Cumulative Flow
# # ne.plot.flow([val_pred[1].squeeze()[::4, ::4, :]], width=5) # one frame
# flw = np.zeros([256, 256, 2])
# flw[..., 0] = cv2.resize(total_flow[..., 0], (256, 256), interpolation=cv2.INTER_NEAREST)
# flw[..., 1] = cv2.resize(total_flow[..., 1], (256, 256), interpolation=cv2.INTER_NEAREST)
# ne.plot.flow([5*flw[::4, ::4, :] / np.max(np.absolute(total_flow))], width=10)
# plt.savefig(prefix + '_flow.png')


# fig, ax = plt.subplots(1,3, figsize=(18,5))
# fig.suptitle(sim_path)

# ax[0].imshow(np.flipud(first_frame), extent=[0, wd, 0, ht], aspect='auto', cmap='gray')
# vert_heat = sns.heatmap(total_flow[...,0], 
#                             mask = mask, 
#                             #vmin=-10, vmax=15, 
#                             center=0, 
#                             ax=ax[0], 
#                             annot=False, 
#                             cmap="vlag", 
#                             #alpha=0.6, 
#                             zorder=2)
# vert_bar = vert_heat.collections[0].colorbar
# vert_bar.ax.tick_params(labelsize=20)
# vert_bar.set_label('Displacement (pixels)', size=20)
# ax[0].axis('off')
# ax[0].set_title('Vertical', fontsize=20)

# ax[1].imshow(np.flipud(first_frame), extent=[0, wd, 0, ht], aspect='auto', cmap='gray')
# hzn_heat = sns.heatmap(total_flow[...,1], 
#                             mask = mask, 
#                             #vmin=-10, vmax=15, 
#                             center=0, 
#                             ax=ax[1], 
#                             annot=False, 
#                             cmap="vlag", 
#                             #alpha=0.6, 
#                             zorder=2)
# hzn_bar = hzn_heat.collections[0].colorbar
# hzn_bar.ax.tick_params(labelsize=20)
# hzn_bar.set_label('Displacement (pixels)', size=20)
# ax[1].axis('off')
# ax[1].set_title('Horizontal', fontsize=20)

# ax[2].imshow(first_frame, cmap='gray', aspect='auto')
# ax[2].set_title("B-Mode", fontsize=20)
# ax[2].axis('off')

# plt.tight_layout()
# plt.savefig(prefix + '_heatmap.png')

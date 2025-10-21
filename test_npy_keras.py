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
    test_inputs, test_outputs = next(test_generator)
    input[i] = test_inputs[0].squeeze() 

    test_pred, test_flow = vxm_model.predict(test_inputs, verbose=0)
    pred[i+1] = test_pred.squeeze()
    hzn_flow[i+1] = test_flow.squeeze()[..., 0]
    vert_flow[i+1] = test_flow.squeeze()[..., 1]
    
print(input.shape, pred.shape, hzn_flow.shape, vert_flow.shape)

# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

render_output(input, pred, hzn_flow, vert_flow)

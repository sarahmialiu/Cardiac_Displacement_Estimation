import os
import numpy as np
import torch
import cv2
import neurite as ne

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import generators

# ------------ MODEL HYPERPARAMETERS AND IMAGE PATHS ---------------

imgs_path = 'Field II Simulation\movement_sequences\\3mm\\0.015_horizontal'
weights_path = 'out\SampleSim_Outputs\model_weights_cropped.pth'    # model weights file path
bidirectional = False                                           # enable bidirectional cost function
batch_size = 1
gpus = [0]
device = 'cuda:0'


# ----------------------- DATA PREPROCESSING -----------------------

# load and prepare testing data
us_files = [imgs_path + "\\" + f for f in os.listdir(imgs_path)]

#Interpolation parameters 
ht=128*6 #768 samples in height direction
wd=3*128
forward_fixed = []
forward_moving = []
backward_fixed = []
backward_moving = []

num_imgs = len(os.listdir(imgs_path))//2
imgs_list = os.listdir(imgs_path)

for img_index, img_name in enumerate(sorted(imgs_list, key=lambda x: int(x.split('_')[1] if x.split('_')[1].isdigit() else 0))):
    if "cropped" in img_name:
        img = cv2.imread(imgs_path + "\\" + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wd, ht), interpolation=cv2.INTER_NEAREST)
        img = cv2.medianBlur(img, 5)

        if img_index > 0 : 
            forward_fixed.append(img / np.max(np.absolute(img)))
            backward_moving.append(img / np.max(np.absolute(img)))
        if img_index < num_imgs-1:
            forward_moving.append(img / np.max(np.absolute(img)))
            backward_fixed.append(img / np.max(np.absolute(img)))

test_fixed = np.array(forward_fixed) # + backward_fixed)
test_moving = np.array(forward_moving) # + backward_moving)

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
        int_downsize=1 # flow downsample factor for integration (default: 2)
    )

model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()
total_flow = np.zeros((wd*2, wd, 2))

for i in range(len(test_fixed)-1):
    test_input, _ = next(test_generator)

    inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in test_input]

    # run inputs through the model to produce a warped image and flow field
    outputs = model(*inputs) # tuple
    pred_img = outputs[0].detach().cpu().numpy()
    pred_flow = outputs[1].permute(0, 2, 3, 1).detach().cpu().numpy()

    total_flow = total_flow + pred_flow[0,:,:,:]


# ----------------------- VISUALIZE MODEL PREDICTIONS -----------------------

# Moving/Fixed/Moved
images = [test_input[0][0,:,:,0], test_input[1][0,:,:,0], pred_img[0,0,:,:], pred_flow[0,:,:,0]]
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

# Cumulative flow
ne.plot.flow([total_flow[::8,::8,:]], width=5)

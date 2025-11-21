import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
from voxelmorph.tf import layers
import tensorflow as tf
import nrrd
from scipy.ndimage import zoom

def resize_3d(data, new_shape=(128,128,128), normalize=False):
    data = data.astype(np.float32)
    X, Y, Z = data.shape
    out = np.zeros(new_shape, dtype=np.float32)

    zoom_factors = (new_shape[0]/X, new_shape[1]/Y, new_shape[2]/Z)
    out = zoom(data, zoom_factors, order=1)  # order=1 = linear

    if normalize:
        out = out.astype(np.float32)
        out = out - out.min()
        if out.max() > 0: out = out / out.max()

    return out

def resize_4d(data, new_shape=(64,64,64), normalize=False):
    data = data.astype(np.float32)
    T, X, Y, Z = data.shape
    out = np.zeros((T, *new_shape), dtype=np.float32)

    zoom_factors = (new_shape[0]/X, new_shape[1]/Y, new_shape[2]/Z)
    for t in range(T):
        out[t] = zoom(data[t], zoom_factors, order=1)  # order=1 = linear

    if normalize:
        out = out.astype(np.float32)
        out = out - out.min()
        if out.max() > 0: out = out / out.max()

    return out

def deform_mask_jump_pairs(mask, hzn_flow, vert_flow, dep_flow):
    mask = resize_3d(mask, new_shape=(128, 128, 128))
    hzn_flow = resize_4d(hzn_flow)
    vert_flow = resize_4d(vert_flow)
    dep_flow = resize_4d(dep_flow)

    warped_mask_dims = [hzn_flow.shape[0], mask.shape[0], mask.shape[1], mask.shape[2]]
    warped_mask = np.zeros(warped_mask_dims)
    transform_layer = layers.SpatialTransformer(interp_method="linear", indexing="ij")
    
    for frame_num in range(1, hzn_flow.shape[0]):
        if frame_num == 1:
            warped_mask[0] = mask
        mask_frame = tf.convert_to_tensor(
            mask[np.newaxis, ..., np.newaxis], 
            dtype=tf.float32)
            
        hzn_flow_fr = hzn_flow[frame_num-1, np.newaxis, ..., np.newaxis]
        vert_flow_fr = vert_flow[frame_num-1, np.newaxis, ..., np.newaxis]
        dep_flow_fr = dep_flow[frame_num-1, np.newaxis, ..., np.newaxis]

        flow_frame = tf.convert_to_tensor(
            np.concatenate([dep_flow_fr, vert_flow_fr, hzn_flow_fr], axis=-1),
            dtype=tf.float32)
        flow_frame = layers.RescaleTransform(2)(flow_frame)
        frame = transform_layer([mask_frame, flow_frame])
        warped_mask[frame_num] = (frame>0.5).numpy().squeeze()

    return warped_mask

def GUI(us, mask, vxm_warpedmask, opt_warpedmask):
    # Default orientation and indices
    vxm_warpedmask = resize_4d(vxm_warpedmask, new_shape=(128, 128, 128))
    us = resize_4d(us, new_shape=(128, 128, 128))
    mask = resize_3d(mask, new_shape=(128, 128, 128))

    orientation = "X, Y"
    slice_index = vxm_warpedmask.shape[3] // 2
    time_index = 0

    # --- Figure setup ---
    fig, ((real_ax, us_ax), (opt_ax, vxm_ax)) = plt.subplots(2, 2)
    plt.subplots_adjust(left=0.2, bottom=0.2)

    real_mask = real_ax.imshow(mask[:, :, slice_index], cmap="gray")
    real_ax.set_title(f"Real Mask (Frame 0)")
    real_ax.set_axis_off()    

    us_img = us_ax.imshow(us[time_index, :, :, slice_index], cmap="gray")
    us_ax.set_title("Ultrasound Scan")
    us_ax.set_axis_off()

    vxm_mask = vxm_ax.imshow(vxm_warpedmask[time_index, :, :, slice_index], cmap="gray")
    vxm_ax.set_title(f"VoxelMorph-Deformed Mask")
    vxm_ax.set_axis_off()

    opt_mask = opt_ax.imshow(opt_warpedmask[time_index, :, :, slice_index], cmap="gray")
    opt_ax.set_title(f"Optical Flow-Deformed Mask")
    opt_ax.set_axis_off()

    fig.suptitle(f"Masked Training Data. Slice: {slice_index}, Frame: {time_index}")

    # --- Slider for slice and time indices ---
    
    slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=slice_ax_slider,
        label="Slice",
        valmin=0,
        valmax=vxm_warpedmask.shape[3] - 1,
        valinit=slice_index,
        valstep=1,
    )

    time_ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_ax_slider,
        label="Time",
        valmin=0,
        valmax=vxm_warpedmask.shape[0] - 1,
        valinit=time_index,
        valstep=1,
    )

    # --- Radio buttons for orientation ---
    ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ("X, Y", "X, Z", "Y, Z"))

    # --- Update functions ---
    def update_display():
        time_idx = int(time_slider.val)
        slice_idx = int(slice_slider.val)
        fig.suptitle(f"Masked Training Data. Slice: {slice_idx}, Frame: {time_idx}")

        if orientation == "X, Y":
            real_data = mask[:, :, slice_idx]
            us_data = us[time_idx, :, :, slice_idx]
            vxm_data = vxm_warpedmask[time_idx, :, :, slice_idx]
            opt_data = opt_warpedmask[time_idx, :, :, slice_idx]

        elif orientation == "X, Z":
            real_data = mask[:, slice_idx, :].T
            us_data = us[time_idx, :, slice_idx, :].T
            vxm_data = vxm_warpedmask[time_idx, :, slice_idx, :].T
            opt_data = opt_warpedmask[time_idx, :, slice_idx, :].T
            
        elif orientation == "Y, Z":
            real_data = mask[slice_idx, :, :].T
            us_data = us[time_idx, slice_idx, :, :].T
            vxm_data = vxm_warpedmask[time_idx, slice_idx, :, :].T
            opt_data = opt_warpedmask[time_idx, slice_idx, :, :].T

        real_mask.set_data(real_data)
        vxm_mask.set_data(vxm_data)
        opt_mask.set_data(opt_data)
        us_img.set_data(us_data)
        fig.canvas.draw_idle()

    def update_image(val):
        update_display()

    def update_orientation(label):
        nonlocal orientation
        orientation = label

        # Update slider range based on orientation
        if orientation == "X, Y":
            slice_slider.valmax = mask.shape[3] - 1
        elif orientation == "X, Z":
            slice_slider.valmax = mask.shape[2] - 1
        elif orientation == "Y, Z":
            slice_slider.valmax = mask.shape[1] - 1

        slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
        slice_slider.set_val(slice_slider.val)  # trigger update
        update_display()

    def on_key(event):
        val = time_slider.val
        if event.key == 'right':
            val = min(time_slider.valmax, val + 1)
        elif event.key == 'left':
            val = max(time_slider.valmin, val - 1)
        time_slider.set_val(val)

    # --- Connect events ---
    time_slider.on_changed(update_image)
    slice_slider.on_changed(update_image)
    radio.on_clicked(update_orientation)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# Load volume

import nrrd

def load_mask_nrrd(path):
    mask, metadata = nrrd.read(path, index_order="C")

    segments = []
    for name in metadata.keys():
        if name.startswith("Segment"):
            if name.endswith("_Name"):
                segments.append(name.split("_")[0])
    segment = segments[-1]
    print(f"Loading mask '{metadata[f'{segment}_Name']}'")
    
    if (mask.ndim != 3):
        layer = int(metadata[f"{segment}_Layer"])
        value = int(metadata[f"{segment}_LabelValue"])
        mask = mask[..., layer]
        mask = np.equal(mask, value)
    return mask

mask = load_mask_nrrd("/home/sarahl/Documents/Fall Rotation/voxelmorph/Optical Flow from Jan Lebert/2025-07-07_US04/2025-07-07_US04_ref0.seg.nrrd")
us = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/Optical Flow from Jan Lebert/2025-07-07_US04/2025-07-07_US04.npy")

# hzn_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Compare_Optical_Vxm_2025-07-07/Viren2d_Flows_refFrame_Predicted/hzn_flow_refFrame.npy") 
# vert_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Compare_Optical_Vxm_2025-07-07/Viren2d_Flows_refFrame_Predicted/vert_flow_refFrame.npy")
# dep_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Compare_Optical_Vxm_2025-07-07/Viren2d_Flows_refFrame_Predicted/dep_flow_refFrame.npy")

# optical_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/Optical Flow from Jan Lebert/2025-07-07_US04/2025-07-07_US04_flows_ref_0.npy")
# hzn_flow = optical_flow[:30, 0, ...]
# vert_flow = optical_flow[:30, 1, ...]
# dep_flow = optical_flow[:30, 2, ...]

# msk = deform_mask_jump_pairs(mask, hzn_flow, vert_flow, dep_flow)
# np.save("warpedmask.npy", msk)

vxm_msk = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Compare_Optical_Vxm_2025-07-07/Viren2d_Flows_refFrame_Predicted/warpedmask.npy")
opt_msk = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Compare_Optical_Vxm_2025-07-07/Viren2d_Flows_refFrame_Optical/warpedmask.npy")
GUI(us, mask, vxm_msk, opt_msk)
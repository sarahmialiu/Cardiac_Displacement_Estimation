import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
from voxelmorph.tf import layers
import tensorflow as tf

def deform_mask(mask, hzn_flow, vert_flow, dep_flow):
    warped_mask = np.zeros(mask.shape)
    residual = np.zeros(mask.shape)
    transform_layer = layers.SpatialTransformer(interp_method="linear", indexing="ij")
    
    for frame_num in range(1, mask.shape[0]):
        if frame_num == 1:
            warped_mask[0] = mask[0]
            mask_frame = tf.convert_to_tensor(
                mask[0, np.newaxis, ..., np.newaxis], 
                dtype=tf.float32)
        else:
            mask_frame = tf.convert_to_tensor(
                warped_mask[frame_num-1, np.newaxis, ..., np.newaxis],
                dtype=tf.float32)
            
        hzn_flow_fr = hzn_flow[frame_num-1, np.newaxis, ..., np.newaxis]
        vert_flow_fr = vert_flow[frame_num-1, np.newaxis, ..., np.newaxis]
        dep_flow_fr = dep_flow[frame_num-1, np.newaxis, ..., np.newaxis]

        # flow_frame = tf.convert_to_tensor(
        #     np.concatenate([hzn_flow_fr, vert_flow_fr, dep_flow_fr], axis=-1), 
        #     dtype=tf.float32)
        flow_frame = tf.convert_to_tensor(
            np.concatenate([dep_flow_fr, vert_flow_fr, hzn_flow_fr], axis=-1),
            dtype=tf.float32)
        flow_frame = layers.RescaleTransform(2)(flow_frame)
        frame = transform_layer([mask_frame, flow_frame])
        warped_mask[frame_num] = frame.numpy().squeeze()

        residual[frame_num] = - mask[frame_num] + warped_mask[frame_num]
    
    # Default orientation and indices
    orientation = "X, Y"
    slice_index = mask.shape[3] // 2
    time_index = mask.shape[0] // 2

    # --- Figure setup ---
    fig, (real_ax, pred_ax, res_ax) = plt.subplots(1, 3)
    plt.subplots_adjust(left=0.2, bottom=0.2)

    real_mask = real_ax.imshow(mask[time_index, :, :, slice_index], cmap="gray")
    real_ax.set_title(f"Real Mask")
    real_ax.set_axis_off()    

    pred_mask = pred_ax.imshow(warped_mask[time_index, :, :, slice_index], cmap="gray")
    pred_ax.set_title(f"Deformed Mask")
    pred_ax.set_axis_off()

    res_mask = res_ax.imshow(residual[time_index, :, :, slice_index], cmap="gray")
    res_ax.set_title("Residual Plot")
    res_ax.set_axis_off()

    fig.suptitle(f"Slice: {slice_index}, Frame: {time_index}")

    # --- Slider for slice and time indices ---
    
    slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=slice_ax_slider,
        label="Slice",
        valmin=0,
        valmax=mask.shape[3] - 1,
        valinit=slice_index,
        valstep=1,
    )

    time_ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_ax_slider,
        label="Time",
        valmin=0,
        valmax=mask.shape[0] - 1,
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
        fig.suptitle(f"Slice: {slice_idx}, Frame: {time_idx}")

        if orientation == "X, Y":
            real_data = mask[time_idx, :, :, slice_idx]
            pred_data = warped_mask[time_idx, :, :, slice_idx]
            res_data = residual[time_idx, :, :, slice_idx]
        elif orientation == "X, Z":
            real_data = mask[time_idx, :, slice_idx, :].T
            pred_data = warped_mask[time_idx, :, slice_idx, :].T
            res_data = residual[time_idx, :, slice_idx, :].T
        elif orientation == "Y, Z":
            real_data = mask[time_idx, slice_idx, :, :].T
            pred_data = warped_mask[time_idx, slice_idx, :, :].T
            res_data = residual[time_idx, slice_idx, :, :].T

        real_mask.set_data(real_data)
        pred_mask.set_data(pred_data)
        res_mask.set_data(res_data)
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
mask = np.load("/home/sarahl/Documents/Fall Rotation/VoxelMorph/resizedmask_random.npy", allow_pickle=True)
hzn_flow = np.load("/home/sarahl/Documents/Fall Rotation/VoxelMorph/hzn_flow_random.npy")
vert_flow = np.load("/home/sarahl/Documents/Fall Rotation/VoxelMorph/vert_flow_random.npy")
dep_flow = np.load("/home/sarahl/Documents/Fall Rotation/VoxelMorph/dep_flow_random.npy")
deform_mask(mask, hzn_flow, vert_flow, dep_flow)
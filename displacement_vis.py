import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from sklearn.metrics import mean_squared_error

def render_output(input, pred, real, hzn_flow, vert_flow, dep_flow):
    """
    Interactive viewer for 4D ultrasound data (T, X, Y, Z).
    
    Args:
        file_path (str): Path to a .npy file containing a 4D numpy array.
    """

    assert input.shape == pred.shape and hzn_flow.shape == vert_flow.shape and vert_flow.shape == dep_flow.shape, \
        f"Trying to visualize images with different shapes. \
            Input: {input.shape}, Pred: {pred.shape}, Horizontal Flow: {hzn_flow.shape}, Vertical Flow: {vert_flow.shape}, Depth Flow: {dep_flow.shape}"

    # Default orientation and indices
    orientation = "X, Y"
    slice_index = input.shape[3] // 2
    time_index = input.shape[0] // 2

    # --- Figure setup ---
    fig, ((input_ax, pred_ax, real_ax), (hzn_flow_ax, vert_flow_ax, vector_ax)) = plt.subplots(2, 3)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    input_mse = mean_squared_error(input[time_index, :, :, slice_index].ravel(), pred[time_index, :, :, slice_index].ravel())
    pred_mse = mean_squared_error(pred[time_index, :, :, slice_index].ravel(), real[time_index, :, :, slice_index].ravel())
    fig.suptitle(f"MSE Input: {input_mse:.3e}       Slice: {slice_index}       MSE Real: {pred_mse:.3e}")
    
    input_img = input_ax.imshow(input[time_index, :, :, slice_index], cmap="gray")
    input_ax.set_title(f"Input (Frame: {time_index})")
    input_ax.set_axis_off()

    pred_img = pred_ax.imshow(pred[time_index, :, :, slice_index], cmap="gray")
    pred_ax.set_title(f"Predicted (Frame: {time_index+1})")
    pred_ax.set_axis_off()

    real_img = real_ax.imshow(real[time_index, :, :, slice_index], cmap='gray')
    real_ax.set_title(f"Real (Frame: {time_index+1})")
    real_ax.set_axis_off()

    hzn_flow_img = hzn_flow_ax.imshow(hzn_flow[time_index, :, :, slice_index//2], cmap="bwr", vmin = -1, vmax = 1)
    hzn_flow_ax.set_title(f"Horizontal Displacement \nFrames {time_index}-{time_index+1}")
    hzn_flow_ax.set_axis_off()
    
    vert_flow_img = vert_flow_ax.imshow(vert_flow[time_index, :, :, slice_index//2], cmap="bwr", vmin = -1, vmax = 1)
    vert_flow_ax.set_title(f"Vertical Displacement \nFrames {time_index}-{time_index+1}")
    vert_flow_ax.set_axis_off()
    
    vector_img  = vector_ax.imshow(input[time_index, :, :, slice_index], cmap="gray")
    Y, X = np.mgrid[0:128:2, 0:128:2]
    quiver = vector_ax.quiver(X, Y, 5*hzn_flow[time_index, :, :, slice_index//2], 5*vert_flow[time_index, :, :, slice_index//2], color='red', scale=60)
    vector_ax.set_title(f"Displacement \nFrames {time_index}-{time_index+1}")

    fig.colorbar(hzn_flow_img, ax=hzn_flow_ax, orientation='horizontal')
    fig.colorbar(vert_flow_img, ax=vert_flow_ax, orientation='vertical')

    # --- Slider for slice and time indices ---
    slice_ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=slice_ax_slider,
        label="Slice",
        valmin=0,
        valmax=input.shape[3] - 1,
        valinit=slice_index,
        valstep=1,
    )

    time_ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=time_ax_slider,
        label="Time",
        valmin=0,
        valmax=input.shape[0] - 1,
        valinit=time_index,
        valstep=1,
    )

    # --- Radio buttons for orientation ---
    ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ("X, Y", "X, Z", "Y, Z"))

    # --- Update functions ---
    def update_display():
        slice_idx = int(slice_slider.val)
        time_idx = int(time_slider.val)

        if orientation == "X, Y":
            input_data = input[time_idx, :, :, slice_idx]
            pred_data = pred[time_idx, :, :, slice_idx]
            real_data = real[time_idx, :, :, slice_idx]
            hzn_flow_data = hzn_flow[time_idx, :, :, slice_idx//2]
            vert_flow_data = vert_flow[time_idx, :, :, slice_idx//2]
        elif orientation == "X, Z":
            input_data = input[time_idx, :, slice_idx, :].T
            pred_data = pred[time_idx, :, slice_idx, :].T
            real_data = real[time_idx, :, slice_idx, :].T
            hzn_flow_data = hzn_flow[time_idx, :, slice_idx//2, :].T
            vert_flow_data = vert_flow[time_idx, :, slice_idx//2, :].T
        elif orientation == "Y, Z":
            input_data = input[time_idx, slice_idx, :, :].T
            pred_data = pred[time_idx, slice_idx, :, :].T
            real_data = real[time_idx, slice_idx, :, :].T
            vert_flow_data = vert_flow[time_idx, slice_idx//2, :, :].T
            hzn_flow_data = hzn_flow[time_idx, slice_idx//2, :, :].T

        input_img.set_data(input_data)
        pred_img.set_data(pred_data)
        real_img.set_data(real_data)
        hzn_flow_img.set_data(hzn_flow_data)
        vert_flow_img.set_data(vert_flow_data)
        vector_img.set_data(input_data)
        quiver.set_UVC(5*hzn_flow_data, 5*vert_flow_data)

        input_ax.set_title(f"Input (Frame: {time_idx})")
        pred_ax.set_title(f"Predicted (Frame: {time_idx+1})")
        real_ax.set_title(f"Real (Frame: {time_idx+1})")

        hzn_flow_ax.set_title(f"Horizontal Displacement \nFrames {time_idx}-{time_idx+1}")
        vert_flow_ax.set_title(f"Vertical Displacement \nFrames {time_idx}-{time_idx+1}")
        vector_ax.set_title(f"Displacement \nFrames {time_idx}-{time_idx+1}")    
        
        in_mse = mean_squared_error(input_data.ravel(), pred_data.ravel())
        rl_mse = mean_squared_error(pred_data.ravel(), real_data.ravel())
        fig.suptitle(f"MSE Input: {in_mse:.3e}      Slice: {slice_idx}      MSE Real: {rl_mse:.3e}")
        fig.canvas.draw_idle()

    def update_image(val):
        update_display()

    def update_orientation(label):
        nonlocal orientation
        orientation = label

        # Update slider range based on orientation
        if orientation == "X, Y":
            slice_slider.valmax = input.shape[3] - 1
        elif orientation == "X, Z":
            slice_slider.valmax = input.shape[2] - 1
        elif orientation == "Y, Z":
            slice_slider.valmax = input.shape[1] - 1

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
    slice_slider.on_changed(update_image)
    time_slider.on_changed(update_image)
    radio.on_clicked(update_orientation)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# Load volume
# input = np.load("/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30.npy", allow_pickle=True)
# pred = np.load("/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30_biv.npy", allow_pickle=True)
# # mask = np.load("/home/sarahl/Documents/Fall Rotation/VoxelMorph/resizedmask.npy", allow_pickle=True)
# hzn_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Masked/RefFrame/hzn_flow_jump.npy")
# vert_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Masked/RefFrame/vert_flow_jump.npy")
# dep_flow = np.load("/home/sarahl/Documents/Fall Rotation/voxelmorph/out/Masked/RefFrame/dep_flow_jump.npy")
# # render_output(input, pred, hzn_flow, vert_flow, dep_flow)
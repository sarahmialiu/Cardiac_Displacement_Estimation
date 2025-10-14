import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def render_output(input, pred, hzn_flow, vert_flow):
    """
    Interactive viewer for 4D ultrasound data (T, X, Y, Z).
    
    Args:
        file_path (str): Path to a .npy file containing a 4D numpy array.
    """

    assert input.shape == pred.shape and pred.shape == flow.shape, \
        f"Trying to visualize images with different shapes. \
            Input: {input.shape}, Pred: {pred.shape}, Flow: {flow.shape}"

    # Default orientation and indices
    orientation = "X, Y"
    slice_index = input.shape[3] // 2
    time_index = input.shape[0] // 2

    # --- Figure setup ---
    fig, (input_ax, pred_ax, hzn_flow_ax, vert_flow_ax) = plt.subplots(1, 3, figsize=(20, 5))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    input_img = input_ax.imshow(input[time_index, :, :, slice_index], cmap="gray")
    input_ax.set_title(f"Input ({orientation}), Slice: {slice_index}, Time: {time_index}")

    pred_img = pred_ax.imshow(pred[time_index, :, :, slice_index], cmap="gray")
    pred_ax.set_title(f"Predicted ({orientation}), Slice: {slice_index}, Time: {time_index}")
    
    hzn_flow_img = hzn_flow_ax.imshow(flow[time_index, :, :, slice_index], cmap="gray")
    hzn_flow_ax.set_title(f"Horizontal Displacement ({orientation}), Slice: {slice_index}, Time: {time_index}")

    vert_flow_img = vert_flow_ax.imshow(flow[time_index, :, :, slice_index], cmap="gray")
    vert_flow_ax.set_title(f"Vertical Displacement ({orientation}), Slice: {slice_index}, Time: {time_index}")

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
            hzn_flow_data = hzn_flow[time_idx, :, :, slice_idx]
            vert_flow_data = vert_flow[time_idx, :, :, slice_idx]
        elif orientation == "X, Z":
            input_data = input[time_idx, :, slice_idx, :].T
            pred_data = pred[time_idx, :, slice_idx, :].T
            hzn_flow_data = hzn_flow[time_idx, :, slice_idx, :].T
            vert_flow_data = vert_flow[time_idx, :, slice_idx, :].T
        elif orientation == "Y, Z":
            input_data = input[time_idx, slice_idx, :, :].T
            pred_data = pred[time_idx, slice_idx, :, :].T
            vert_flow_data = vert_flow[time_idx, slice_idx, :, :].T
            hzn_flow_data = hzn_flow[time_idx, slice_idx, :, :].T

        input_img.set_data(input_data)
        pred_img.set_data(pred_data)
        hzn_flow_img.set_data(hzn_flow_data)
        vert_flow_img.set_data(vert_flow_data)
        input_ax.set_title(f"Input ({orientation}), Slice: {slice_idx}, Time: {time_idx}")
        pred_ax.set_title(f"Predicted ({orientation}), Slice: {slice_idx}, Time: {time_idx}")
        hzn_flow_ax.set_title(f"Horizontal Displacement ({orientation}), Slice: {slice_idx}, Time: {time_idx}")
        vert_flow_ax.set_title(f"Vertical Displacement ({orientation}), Slice: {slice_idx}, Time: {time_idx}")
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

    # --- Connect events ---
    slice_slider.on_changed(update_image)
    time_slider.on_changed(update_image)
    radio.on_clicked(update_orientation)

    plt.show()

# Load volume
# input = np.load("/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30.npy", allow_pickle=True)
# pred = np.load("/home/sarahl/Documents/Fall Rotation/DataVisualization/data/ultrasound_4D_npy/2024-06-26_US30_biv.npy", allow_pickle=True)

# render_output(input, pred)
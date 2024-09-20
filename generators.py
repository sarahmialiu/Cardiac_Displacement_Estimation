import numpy as np

def custom_generator(x_data1, x_data2, batch_size=8):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data1.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data1.shape[0], size=batch_size)
        moving_images = x_data1[idx1, ..., np.newaxis]
        fixed_images = x_data2[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)
        
def ordered_generator(x_data1, x_data2, batch_size=8):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data1.shape[1:] # extract data shape
    ndims = len(vol_shape)

    num_samples = x_data1.shape[0]
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    start_idx = 0
    
    while True:
        if start_idx + batch_size > num_samples:
            start_idx = 0

        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.arange(start_idx, start_idx + batch_size) % num_samples
        moving_images = x_data1[idx1, ..., np.newaxis]
        fixed_images = x_data2[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image) to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]

        start_idx += batch_size
        
        yield (inputs, outputs)

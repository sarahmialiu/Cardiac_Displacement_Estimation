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
        
def vol_generator(x_data1, x_data2, batch_size=8):
    """
    Generator that takes in data of size [N, H, W, D], and yields data for
    our custom 3d vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, D, 1], fixed image [bs, H, W, D, 1]
    outputs: moved image [bs, H, W, D, 1], zero-gradient [bs, H, W, D, 2]
    """

    # preliminary sizing
    vol_shape = x_data1.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, D, 1]
        idx1 = np.random.randint(0, x_data1.shape[0], size=batch_size)
        moving_images = x_data1[idx1, ..., np.newaxis]
        fixed_images = x_data2[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (tuple(inputs), tuple(outputs))


def ordered_vol_generator_jump_pairs(x_data1, x_data2, batch_size=8):
    """
    Generator that yields pairs where the moving image is always the first frame (index 0),
    and the fixed images are all other frames in order.

    Parameters
    ----------
    x_data1 : np.ndarray
        Moving image data, shape [N, H, W, D].
    x_data2 : np.ndarray
        Fixed image data, shape [N, H, W, D].
    batch_size : int
        Number of samples per batch.

    Yields
    ------
    idx_batch : np.ndarray
        The indices of the fixed images in this batch.
    inputs : tuple of np.ndarray
        ([moving_images, fixed_images])
    outputs : tuple of np.ndarray
        ([fixed_images, zero_phi])
    """

    # preliminary sizing
    vol_shape = x_data1.shape[1:]
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
        # images need to be of the size [batch_size, H, W, D, 1]
        idx1 = np.arange(start_idx, start_idx + batch_size) % num_samples
        moving_image = x_data1[0, ..., np.newaxis]
        moving_images = np.repeat(moving_image[np.newaxis, ...], batch_size, axis=0)
        fixed_images = x_data2[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]

        start_idx += batch_size
        
        yield (idx1, tuple(inputs), tuple(outputs))

def vol_generator_input_output(x_data1, x_data2, batch_size=8):
    """
    Generator that takes in data of size [N, H, W, D], and yields data for
    our custom 3d vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, D, 1], fixed image [bs, H, W, D, 1]
    outputs: moved image [bs, H, W, D, 1], zero-gradient [bs, H, W, D, 2]
    """

    # preliminary sizing
    vol_shape = x_data1.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, D, 1]
        idx1 = np.random.randint(0, x_data1.shape[0], size=batch_size)
        moving_images = x_data1[idx1, ..., np.newaxis]
        fixed_images = x_data2[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [np.concatenate([fixed_images, moving_images], axis=-1), zero_phi]
        
        yield (tuple(inputs), tuple(outputs))

def ordered_vol_generator(x_data1, x_data2, batch_size=1):
    """
    Generator that takes in data of size [N, H, W, D], and yields data for
    our custom 3d vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, D, 1], fixed image [bs, H, W, D, 1]
    outputs: moved image [bs, H, W, D, 1], zero-gradient [bs, H, W, D, 2]
    """

    # preliminary sizing
    vol_shape = x_data1.shape[1:]
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
        # images need to be of the size [batch_size, H, W, D, 1]
        idx1 = np.arange(start_idx, start_idx + batch_size) % num_samples
        moving_images = x_data1[idx1, ..., np.newaxis]
        fixed_images = x_data2[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]

        start_idx += batch_size
        
        yield (idx1, tuple(inputs), tuple(outputs))

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

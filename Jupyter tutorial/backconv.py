def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ############################################################################
    #                      START HELPER FUNCTIONS                              #
    ############################################################################

    ############################################################################
    #                      START HELPER FUNCTIONS                              #
    ############################################################################
    def pad_3D(x, pad): 

        # Some important dimensions
        c, h, w = np.shape(x)
        
        # Initialize padded_x
        x_padded = np.zeros((c, h + 2 * pad, h + 2 * pad))

        # Clone original x to internal part of x_padded 
        for chan in range(c):
            for down in range(h):
                for right in range(w):
                    x_padded[chan][down + pad][right + pad] = x[chan][down][right]      

        return x_padded

    def get_spatial_score(x, f, b):
        score = np.sum(x * f) + b
        return score

    def get_spatial_slice(x, f, right, down, stride):

        # Get dimensions from current weight f in w
        cc, hh, ww = np.shape(f) 
        
        # Getting x's spacial slice
        right_init = right * stride
        right_end  = right_init + ww
            
        down_init  = down  * stride 
        down_end   = down_init + hh
            
        x_spatial = x[:, down_init:down_end, right_init:right_end]
        
        return x_spatial

    def fil2col(w):
        F, c, h, w_ = np.shape(w)
        rows = F
        cols = c * h * w_
        
        f_matrix = np.zeros((rows, cols))
        
        for f in range(F):
            f_matrix[f] = np.reshape(w[f], cols)
            
        return f_matrix
    
    def im2col(x, f, stride):
        
        # Get dimensions from current weight f in w
        cc, hh, ww, out_h, out_w = f
        
        # Compute Shape of im2col
        rows = cc * hh * ww
        cols = out_h * out_h
        
        # Create current xn im2col
        xn_im2col = np.zeros((rows, cols))

        location = 0
        for down in range(out_h):
            for right in range(out_w):
                        
                # Getting x's spacial slice
                right_init = right * stride
                right_end  = right_init + ww
            
                down_init  = down  * stride 
                down_end   = down_init + hh
            
                x_spatial = x[:, down_init:down_end, right_init:right_end]
        
                col_size = cc * hh * ww
                col = np.reshape(x_spatial, col_size)
                        
                xn_im2col[:, location] = col
                location += 1
          
        return xn_im2col

    x, w, b, conv_param = cache

    def conv_backward_unvectorized(x, w, b, conv_paramm, dout):    
    
        # Some important quantities:
        stride = conv_param['stride']
        pad    = conv_param['pad']
        
        # Get params of x:
        N, c, h, w_ = np.shape(x)
                
        # Get params of filters (w):
        F, cc, hh, ww = np.shape(w)

        # Get params of biases (b):
        ff, = np.shape(b)        
        
        # Compute spatial size
        out_h = int(1 + (h  + 2 * pad - hh) / stride)
        out_w = int(1 + (w_ + 2 * pad - ww) / stride)

        # Create output volumes
        dx = np.zeros((N, F, out_h, out_w))
        dw = np.zeros((F, cc, hh, ww))
        db = np.zeros((F, cc, hh, ww))

        
        # Traverse all training examples ----------------------
        for n in range(N):
            
            # Pad current x[n]
            dxn_padded = pad_3D(x[n], pad)

            # Traverse all filters -----------------------------------------------------------
            for f in range(F):     

                # Convolute 
                for down in range(out_h):
                    for right in range(out_w):

                        # Getting spacial slice
                        right_init = right * stride
                        right_end  = right_init + ww
            
                        down_init  = down  * stride 
                        down_end   = down_init + hh

                        dw[f] = dw[f] + dxn_padded[N, :, right_init : right_end, down_init : down_end] * dout[n, f, down, right]
                        dxn_padded[n, :, right_init : right_end, down_init : down_end] = dxn_padded[n, :, right_init : right_end, down_init : down_end] +  w[f] * dout[n, f, down, right_end]
                        
        dx = dxn_padded[:, :, pad : pad + hh, pad : pad + ww]

        return dx, dw, db

    ############################################################################
    #                        END HELPER FUNCTIONS                              #
    ############################################################################

    dx, dw, db = conv_backward_unvectorized(x, w, b, conv_param, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

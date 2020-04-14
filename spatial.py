import numpy as np

def spatial_scale_conv_3x3_stride_1(s, p):
    """
    This method computes the spatial scale of 3x3 convolutional layer with stride 1 in terms of its input feature map's
    spatial scale value (s) and spatial overal value (p).
    """
    
    return s + 4 * (1-p) * s + 4 * (1-p) ** 2 * s

def spatial_overlap_conv_3x3_stride_1(p):
    """
    This method computes the spatial overlap of 3x3 convolutional layer with stride 1 in terms of its input feature map's
    spatial overal value (p).
    """

    return (1 + (1-p) * (5-2*p))/(1+4 * (1-p) *(2-p))

def spatial_scale_conv_3x3_stride_1_pooling_2x2_stride_2(s, p):
    """
    This method computes the combined spatial scale of 3x3 convolutional layer with stride 1 followed by a 2x2 pooling layer
    with stride 2 in terms of their input feature map's spatial scale value (s) and spatial overal value (p).
    """

    s_t, p_t = spatial_scale_conv_3x3_stride_1(s, p), spatial_overlap_conv_3x3_stride_1(p)
    return spatial_scale_pooling_2x2_stride_2(s_t, p_t)

def spatial_overlap_conv_3x3_stride_1_pooling_2x2_stride_2(p):
    """
    This method computes the combined spatial overlap of 3x3 convolutional layer with stride 1 followed by a 2x2 pooling layer
    in terms of their input feature map's spatial overal value (p).
    """

    return spatial_overlap_pooling_2x2_stride_2(spatial_overlap_conv_3x3_stride_1(p))

def spatial_scale_pooling_2x2_stride_2(s, p):
    """
    This method computes the spatial scale of 2x2 pooling layer with stride 2 in terms of its input feature map's
    spatial scale value (s) and spatial overal value (p).
    """

    return (2-p) ** 2 * s

def spatial_overlap_pooling_2x2_stride_2(p):
    """
    This method computes the spatial overlap of 2x2 pooling layer with stride 2 in terms of its input feature map's
    spatial overal value (p).
    """

    return p/(2-p)

def spatial_scale_conv_3x3_stride_2(s, p):
    """
    This method computes the spatial scale of 3x3 convolutional layer with stride 2 in terms of its input feature map's
    spatial scale value (s) and spatial overal value (p).
    """

    return spatial_scale_conv_3x3_stride_1(s, p)

def spatial_overlap_conv_3x3_stride_2(p):
    """
    This method computes the spatial overlap of 3x3 convolutional layer with stride 2 in terms of its input feature map's
    spatial overal value (p).
    """

    return (1 + 2 * (1-p)) / (1 + 4 * (1 - p) * (2 - p))

def spatial_scale_pooling_3x3_stride_2(s, p):
    """
    This method computes the spatial scale of 3x3 pooling layer with stride 1 in terms of its input feature map's
    spatial scale value (s) and spatial overal value (p).
    """

    return spatial_scale_conv_3x3_stride_1(s, p)

def spatial_overlap_pooling_3x3_stride_2(p):
    """
    This method computes the spatial overlap of 3x3 convolutional layer with stride 2 in terms of its input feature map's
    spatial overal value (p).
    """

    return spatial_overlap_conv_3x3_stride_2(p)

def spatial_scale_conv_7x7_stride_2(s, p):
    """
    This method computes the spatial scale of 7x7 convolutional layer with stride 2 assuming it is the first layer of the 
    network. In other words, this method's ouptut is incorrect if this layer is not the first layer of the network. 
    """

    return 7 * 7

def spatial_overlap_conv_7x7_stride_2(p):
    """
    This method computes the spatial overlap of 7x7 convolutional layer with stride 2 assuming it is the first layer of the 
    network. In other words, this method's ouptut is incorrect if this layer is not the first layer of the network. 
    """

    return 5.0 / 7

def spatial_scale_conv_1x1_stride_2(s, p):
    """
    This method computes the spatial scale of 1x1 convolutional layer with stride 2 in terms of its input feature map's
    spatial scale value (s) and spatial overal value (p).
    """

    return s

def spatial_overlap_conv_1x1_stride_2(p):
    """
    This method computes the spatial overlap of 1x1 convolutional layer with stride 2 in terms of its input feature map's
    spatial overal value (p).
    """

    return 2 * max(p - 0.5, 0)

def spatial_scale_conv_1x1_stride_1(s, p):
    """
    This method computes the spatial scale of 1x1 convolutional layer with stride 1 in terms of its input feature map's
    spatial scale value (s) and spatial overal value (p).
    """

    return s

def spatial_overlap_conv_1x1_stride_1(p):
    """
    This method computes the spatial overlap of 1x1 convolutional layer with stride 1 in terms of its input feature map's
    spatial overal value (p).
    """

    return p

def spatial_scale_conv_3x3_stride_1_dilate_2(s, p):
    """
    This method computes the spatial scale of dilated 3x3 convolutional layer with stride 1 and dilation rate of 2 in terms 
    of its input feature map's spatial scale value (s) and spatial overal value (p).
    """

    return 9 * s - 24 * max(p - 0.5, 0) * s

def spatial_overlap_conv_3x3_stride_1_dilate_2(p):
    """
    This method computes the spatial overlap of dilated 3x3 convolutional layer with stride 1 and dialation rate of 2 in terms
    of its input feature map's spatial overal value (p).
    """

    denominator = 9 - 24 * max(p - 0.5, 0)
    if p < 0.5:
        return 15 * p / denominator
    else:
        return (6 + 3 * (1 - p) - (14 + 4 * (1 - p)) * max(p - 0.5, 0)) / denominator

#The mapping from the layer names to their corresponding fns that compute their spatial scales.
spatial_scales = {'conv_3x3_stride_1': spatial_scale_conv_3x3_stride_1,
                  'conv_3x3_stride_1_dilate_2': spatial_scale_conv_3x3_stride_1_dilate_2,
                  'conv_3x3_stride_1_pooling_2x2_stride_2': spatial_scale_conv_3x3_stride_1_pooling_2x2_stride_2,
                  'conv_3x3_stride_2': spatial_scale_conv_3x3_stride_2,
                  'pooling_3x3_stride_2': spatial_scale_pooling_3x3_stride_2,
                  'conv_7x7_stride_2': spatial_scale_conv_7x7_stride_2,
                  'conv_1x1_stride_2': spatial_scale_conv_1x1_stride_2,
                  'conv_1x1_stride_1': spatial_scale_conv_1x1_stride_1}

#The mapping from the layer names to their corresponding fns that compute their spatial overlaps.
spatial_overlaps = {'conv_3x3_stride_1': spatial_overlap_conv_3x3_stride_1,
                    'conv_3x3_stride_1_dilate_2': spatial_overlap_conv_3x3_stride_1_dilate_2,
                    'conv_3x3_stride_1_pooling_2x2_stride_2': spatial_overlap_conv_3x3_stride_1_pooling_2x2_stride_2,
                    'conv_3x3_stride_2': spatial_overlap_conv_3x3_stride_2,
                    'pooling_3x3_stride_2': spatial_overlap_pooling_3x3_stride_2,
                    'conv_7x7_stride_2': spatial_overlap_conv_7x7_stride_2,
                    'conv_1x1_stride_2': spatial_overlap_conv_1x1_stride_2,
                    'conv_1x1_stride_1': spatial_overlap_conv_1x1_stride_1}

def process(layers):
    """
    This method takes a list of layers and computes their corresponding spatial scales 
    and spatial overlaps. In prticular, it recursively applies the spatial scale and overlap
    formulas of each layer on its previous layer spatial scale and overla.
    """

    #s: spatial scale
    #p: spatial overlap
    s, p = [1], [0]
    #The initial value for spatial scale is equal to 1 since the first feature map is in fact
    #the input image itself and each entry of such a feature map is a pixel. Therefore, the
    #spatial scale of image entries (pixels) is equal to one.
    #On the other hand, the initial value for spatial overlap is equal to 0 given the initial
    #feature map is the input image itself and two neighboring enties (pixels) do not have
    #spatial overlap. 
    
    for layer in layers:
        s.append(spatial_scales[layer](s[-1], p[-1]))
        p.append(spatial_overlaps[layer](p[-1]))
    return s, p

def plot(ss, legends, plt, filename = None):
    """
    This method plots the spatial scale profiles of CNNs in terms of their layer depths.
    In particular, ss is a list of spatial scales profiles where len(ss) is equal to the 
    number of CNNs that we want to plot their spatial scale profiles. 
    legends is a list of strings that denote the legend of each spatial scale profile.
    plt is in instance of matplotlib.pyplot.
    If filename is not None, the ploted curves will be saved with the file name passed 
    as filename argument.
    """

    ymin, ymax = min(ss[0]) ** 0.5, max(ss[0]) ** 0.5
    for s in ss[1:]:
        ymin, ymax = min(ymin, min(s) ** 0.5), max(ymax, max(s) ** 0.5)

    style = ['-o', '-*r', '-+g']
    for i, s in enumerate(ss):
        s = np.array(s) ** 0.5
        j = i % 3
        plt.plot(s, style[j])
    
    plt.ylim(ymin, ymax)
    plt.xlim((0, len(ss[0]) - 1))

    plt.legend(legends)
        
    plt.xlabel('layers')
    plt.ylabel('spatial scale width')

    if filename:
        plt.savefig(filename)
    plt.show()
    

def pooling_vs_no_pooling_case_study():
    """
    This method compares the spatial scale gowth rates of two variant of CNNs with 20 layers of 3x3 convolutional 
    layers with stride 1 where one of the CNNs uses 2x2 pooling layers with stride 2 per every 4 convolutional layers
    whereas the other one do not rely on any pooling operations.
    """
    
    n_layers = 20
    pooling_step = 4
    layers = []
    for i in range(n_layers):
        if i != 0 and i % pooling_step == 0:
            layers.append('conv_3x3_stride_1_pooling_2x2_stride_2')
        else:
            layers.append('conv_3x3_stride_1')
    s_with_pooling, p_with_pooling = process(layers)

    layers = ['conv_3x3_stride_1'] * n_layers
    s_without_pooling, p_without_pooling = process(layers)
    
    import matplotlib.pyplot as plt
    print('max spatial scale with pooling:', s_with_pooling[-1])
    print('max spatial scale without pooling:', s_without_pooling[-1])

    plot([s_with_pooling, s_without_pooling], ['3 x 3 conv layers with 2 x 2 pooling every 4 layers' , '3 x 3 conv layers without pooling'], plt,
         'spatial_scale.png')

def dilated_convolutions_case_study():
    """
    This method compare the spatial scale growth rate of a 5 layer convolutional network consists of dilated 3x3 convolutional layers 
    with stride 1 and dialation rate 2, and its counterparts consists of regualr 3x3 convolutional layers with stride 1 and 2.
    """
    
    n_layers = 5
    layers_dilated_stride_1_dilate_2 = ['conv_3x3_stride_1_dilate_2'] * n_layers
    layers_regular_stride_2 = ['conv_3x3_stride_2'] * n_layers
    layers_regualr_stride_1 = ['conv_3x3_stride_1'] * n_layers
    s_stride_1_dilate_2, _ = process(layers_dilated_stride_1_dilate_2)
    s_stride_2, _ = process(layers_regular_stride_2)
    s_stride_1, _ = process(layers_regualr_stride_1)

    print('max spatial scale with stride 1 and dialtion rate of 2:', s_stride_1_dilate_2[-1] ** 0.5)
    print('max spatial scale with stride 2:', s_stride_2[-1] ** 0.5)
    print('max spatial scale with stride 1:', s_stride_1[-1] ** 0.5)

    import matplotlib.pyplot as plt
    plot([s_stride_1_dilate_2, s_stride_2, s_stride_1], ['conv_3x3_stide_1_dilate_2' , 'conv_3x3_stide_2', 'conv_3x3_stride_1'], plt, 
          filename = 'dilated.png')
    

def resnet_50():
    """
    This method computes and plots the spatial scale profile of ResNet-50.
    """

    #Each residual block of ResNet-50 except conv3_1, conv4_1 and conv5_1 ,consists of 3 convolutional layers where the first layer
    #is 1x1 convolutional with stride 1, the second layer is 3x3 convolutional layer with stride 1 and the third layer is 1x1
    #convolutional layer with stride 1 as well.
    resnet_block = ['conv_1x1_stride_1', 'conv_3x3_stride_1', 'conv_1x1_stride_1']

    #The residual blocks conv3_1, conv4_1 and conv5_1 different from other residual blocks perform pooling operations via their
    #first layers. In particular, their first layer is a 1x1 covolutional layer with stride 2.
    resnet_block_pooling = ['conv_1x1_stride_2', 'conv_3x3_stride_1', 'conv_1x1_stride_1']

    #Here, we construct the layers list which characterizes the configuration of each layer of ResNet-50.
    layers = ['conv_7x7_stride_2', 'pooling_3x3_stride_2'] + resnet_block * 3 + resnet_block_pooling + resnet_block * 3
    layers += resnet_block_pooling + resnet_block * 5 + resnet_block_pooling + resnet_block * 2

    #Below, s and p denote layer-wise spatial scale and spatial overlap of ResNet-50.
    s, p = process(layers)

    print('Final spatial scale width:', s[-1] ** 0.5)
        
    #Finally, we plot spatial scale profiel of ResNet-50.
    import matplotlib.pyplot as plt
    plot([s], ['ResNet-50'], plt, filename = 'resnet-50.png')

if __name__ == '__main__':
    dilated_convolutions_case_study()
    #resnet_50()

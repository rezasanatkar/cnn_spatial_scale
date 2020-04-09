import numpy as np

def spatial_scale_conv_3x3_stride_1(s, p):
    return s + 4 * (1-p) * s + 4 * (1-p) ** 2 * s

def spatial_overlap_conv_3x3_stride_1(p):
    return (1 + (1-p) * (5-2*p))/(1+4 * (1-p) *(2-p))

def spatial_scale_conv_3x3_stride_1_pooling_2x2_stride_2(s, p):
    s_t, p_t = spatial_scale_conv_3x3_stride_1(s, p), spatial_overlap_conv_3x3_stride_1(p)
    return spatial_scale_pooling_2x2_stride_2(s_t, p_t)

def spatial_overlap_conv_3x3_stride_1_pooling_2x2_stride_2(p):
    return spatial_overlap_pooling_2x2_stride_2(spatial_overlap_conv_3x3_stride_1(p))

def spatial_scale_pooling_2x2_stride_2(s, p):
    return (2-p) ** 2 * s

def spatial_overlap_pooling_2x2_stride_2(p):
    return p/(2-p)

def spatial_scale_conv_3x3_stride_2(s, p):
    return spatial_scale_conv_3x3_stride_1(s, p)

def spatial_overlap_conv_3x3_stride_2(p):
    return (1 + 2 * (1-p)) / (1 + 4 * (1 - p) * (2 - p))

def spatial_scale_pooling_3x3_stride_2(s, p):
    return spatial_scale_conv_3x3_stride_1(s, p)

def spatial_overlap_pooling_3x3_stride_2(p):
    return spatial_overlap_conv_3x3_stride_2(p)

def spatial_scale_conv_7x7_stride_2(s, p):
    return 7 * 7

def spatial_overlap_conv_7x7_stride_2(p):
    return 5.0 / 7

def spatial_scale_conv_1x1_stride_2(s, p):
    return s

def spatial_overlap_conv_1x1_stride_2(p):
    return 2 * max(p - 0.5, 0)

def spatial_scale_conv_1x1_stride_1(s, p):
    return s

def spatial_overlap_conv_1x1_stride_1(p):
    return p

def spatial_scale_conv_3x3_stride_1_dilate_2(s, p):
    return 9 * s - 24 * max(p - 0.5, 0) * s

def spatial_overlap_conv_3x3_stride_1_dilate_2(p):
    denominator = 9 - 24 * max(p - 0.5, 0)
    if p < 0.5:
        return 15 * p / denominator
    else:
        return (6 + 3 * (1 - p) - (14 + 4 * (1 - p)) * max(p - 0.5, 0)) / denominator

spatial_scales = {'conv_3x3_stride_1': spatial_scale_conv_3x3_stride_1,
                  'conv_3x3_stride_1_dilate_2': spatial_scale_conv_3x3_stride_1_dilate_2,
                  'conv_3x3_stride_1_pooling_2x2_stride_2': spatial_scale_conv_3x3_stride_1_pooling_2x2_stride_2,
                  'conv_3x3_stride_2': spatial_scale_conv_3x3_stride_2,
                  'pooling_3x3_stride_2': spatial_scale_pooling_3x3_stride_2,
                  'conv_7x7_stride_2': spatial_scale_conv_7x7_stride_2,
                  'conv_1x1_stride_2': spatial_scale_conv_1x1_stride_2,
                  'conv_1x1_stride_1': spatial_scale_conv_1x1_stride_1}

spatial_overlaps = {'conv_3x3_stride_1': spatial_overlap_conv_3x3_stride_1,
                    'conv_3x3_stride_1_dilate_2': spatial_overlap_conv_3x3_stride_1_dilate_2,
                    'conv_3x3_stride_1_pooling_2x2_stride_2': spatial_overlap_conv_3x3_stride_1_pooling_2x2_stride_2,
                    'conv_3x3_stride_2': spatial_overlap_conv_3x3_stride_2,
                    'pooling_3x3_stride_2': spatial_overlap_pooling_3x3_stride_2,
                    'conv_7x7_stride_2': spatial_overlap_conv_7x7_stride_2,
                    'conv_1x1_stride_2': spatial_overlap_conv_1x1_stride_2,
                    'conv_1x1_stride_1': spatial_overlap_conv_1x1_stride_1}

def process(layers):
    s, p = [1], [0]
    for layer in layers:
        s.append(spatial_scales[layer](s[-1], p[-1]))
        p.append(spatial_overlaps[layer](p[-1]))
    return s, p

def plot(ss, legends, plt, filename = None):
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
    

def case_study():
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

def conv_3x3_stride_2_case_study():
    n_layers = 5
    layers_stride_1_dilate_2 = ['conv_3x3_stride_1_dilate_2'] * n_layers
    layers_stride_2 = ['conv_3x3_stride_2'] * n_layers
    layers_stride_1 = ['conv_3x3_stride_1'] * n_layers
    s_stride_1_dilate_2, _ = process(layers_stride_1_dilate_2)
    s_stride_2, _ = process(layers_stride_2)
    s_stride_1, _ = process(layers_stride_1)

    print('max spatial scale with stride 1 and dialte rate of 2:', s_stride_1_dilate_2[-1] ** 0.5)
    print('max spatial scale with stride 2:', s_stride_2[-1] ** 0.5)
    print('max spatial scale with stride 1:', s_stride_1[-1] ** 0.5)

    import matplotlib.pyplot as plt
    plot([s_stride_1_dilate_2, s_stride_2, s_stride_1], ['conv_3x3_stide_1_dilate_2' , 'conv_3x3_stide_2', 'conv_3x3_stride_1'], plt, 
          filename = 'dilated.png')
    
resnet_block = ['conv_1x1_stride_1', 'conv_3x3_stride_1', 'conv_1x1_stride_1']
resnet_block_pooling = ['conv_1x1_stride_2', 'conv_3x3_stride_1', 'conv_1x1_stride_1']

def resnet_50():
    layers = ['conv_7x7_stride_2', 'pooling_3x3_stride_2'] + resnet_block * 3 + resnet_block_pooling + resnet_block * 3
    layers += resnet_block_pooling + resnet_block * 5 + resnet_block_pooling + resnet_block * 2
    s, p = process(layers)
    import matplotlib.pyplot as plt
    print('max spatial scale:', s[-1] ** 0.5)
    plot([s], ['ResNet-50'], plt, filename = 'resnet-50.png')

if __name__ == '__main__':
    case_study()
    #conv_3x3_stride_2_case_study()
    #resnet_50()

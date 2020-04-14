<h3 align="center">
<p>Analysis and Applications of Multi-Scale CNN Feature Maps
</h3>
The goal of this repo and its corresponding blogpost is to provide a deeper insight into the intuitions behind the use cases of multi-scale convolutional
feature maps in the recent proposed CNN architectures for variety of vision tasks. Therefore, this project can be treated as a tutorial to learn more
about how different types of layers impact the spatial scales and receptive fields of feature maps. Also, this repo could be used by those engineers and
researchers that are involved in designing CNN architectures and are tired of blind trial and error of which feature maps to choose from a CNN backbone
to improve the performance of their models, and instead, prefer from the early steps of design process, to match the spatial scale profiles of feature
maps with the object dimensions in training datasets.

<p align="center"> 
<img src="2x2pooling.png">
</p>

<p align="center">
  <img ''>
</p>
<p align="center">
!['test'](2x2pooling.png)
</p>

### Features
The module that implements the functionalities of computing spatial scales and overlaps for different layers is `spatial.py`.
![Image description](resnet-50.png)

![Image description](dilated.png)




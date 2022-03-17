# Low_Light

## 2017

### (PR 2017)LLNet

- LLNet: A deep autoencoder approach to natural low-light image enhancement
- 奠基之作

## 2020

### [(CVPR 2020)Zero-DCE](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

- Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- 

### [(CVPR 2020)Learning to Restore Low-Light Images via Decomposition-and-Enhancement](https://ieeexplore.ieee.org/document/9156446)

- 子主题 2
- 

### [(CVPR 2020 DRBN)](https://ieeexplore.ieee.org/document/9156559)

- From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement
- 

### [(CVPR 2020)STARnet](https://alterzero.github.io/projects/star_cvpr2020.pdf)

- Space-Time-Aware Multi-Resolution Video Enhancement
- 

### [(CVPR 2020)DeepLPF](https://arxiv.org/pdf/2003.13985.pdf)

- DeepLPF: Deep Local Parametric Filters for Image Enhancement
- 

### [(TPAMI 2020)Image-Adaptive-3DLUT](https://ieeexplore.ieee.org/document/9206076)

- Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time
- 

### [(CVPR 2020)Learning temporal consistency for low light video enhancement from single images ](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf)

- 子主题 2
- 

## 2021

### [(CVPR 2021)RUAS ](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf)

- Retinex-Inspired Unrolling with Cooperative Prior Architecture Search for Low-Light Image Enhancement
- 

### [(CVPR 2021)Deep denoising of flash and no-flash pairs for photography in low-light environments ](https://openaccess.thecvf.com/content/CVPR2021/papers/Xia_Deep_Denoising_of_Flash_and_No-Flash_Pairs_for_Photography_in_CVPR_2021_paper.pdf)

- 子主题 2
- 

### [(CVPR 2021)HORUS ](https://openaccess.thecvf.com/content/CVPR2021/papers/Moseley_Extreme_Low-Light_Environment-Driven_Image_Denoising_Over_Permanently_Shadowed_Lunar_Regions_CVPR_2021_paper.pdf)

- Extreme Low-Light Environment-Driven Image Denoising over Permanently Shadowed Lunar Regions with a Physical Noise Model
- 

### [(CVPR 2021)Nighttime visibility enhancement by increasing the dynamic range and suppression of light effects ](https://openaccess.thecvf.com/content/CVPR2021/papers/Sharma_Nighttime_Visibility_Enhancement_by_Increasing_the_Dynamic_Range_and_Suppression_CVPR_2021_paper.pdf)

- 子主题 2
- 

### [(ICCV 2021)SDSD ](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Seeing_Dynamic_Scene_in_the_Dark_A_High-Quality_Video_Dataset_ICCV_2021_paper.pdf)

- Seeing Dynamic Scene in the Dark: A High-Quality Video Dataset with Mechatronic Alignment
- 

### [(ICCV 2021)DeepHDRVideo ](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_HDR_Video_Reconstruction_A_Coarse-To-Fine_Network_and_a_Real-World_Benchmark_ICCV_2021_paper.pdf)

- HDR Video Reconstruction: A Coarse-to-Fine Network and a Real-World Benchmark Dataset
- 

### [(ICCV 2021)MID ](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_Matching_in_the_Dark_A_Dataset_for_Matching_Image_Pairs_ICCV_2021_paper.pdf)

- Matching in the Dark: A Dataset for Matching Image Pairs of Low-Light Scenes
- 

### [(ICCV 2021)UTVNet ](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Adaptive_Unfolding_Total_Variation_Network_for_Low-Light_Image_Enhancement_ICCV_2021_paper.pdf)

- Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement
- 

### [(ICCV 2021)LLVIP ](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/papers/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.pdf)

- LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision
- 

### [(AAAI 2022)LLFLOW](https://arxiv.org/pdf/2109.05923.pdf)

- Low-Light Image Enhancement with Normalizing Flow
- 

## 语义先验

### [(MM 2020)Integrating Semantic Segmentation and Retinex Model for Low Light Image Enhancement](https://mm20-semanticreti.github.io)

- 
- Motivation

	- 语义信息可以为低光照图像增强提供丰富的信息

- Contribution

	- 提出一种结合Retinex模型和深度学习的方法，并且加入了语义先验。
	- 语义先验通过空间变换处理特征来指导光照和反射率的联合增强，从而提高区域恢复的恢复质量。
	- 建立了一种新的低光照图像合成模型，考虑曝光、颜色、噪声等因素。并且收集了100张图像。

- Method

	- 
	- 包含三个模块，四个子网络
	- DecompNet：利用Retinex理论，将input分解为反射图和照度图。用成对的低光和正常光图像训练。
	- SegNet：一个轻量的U-Net，只用来区分sky、ground、foreground objects。
	- Reflectance Enhancement：基于RIR组成，结合语义特征，设计了一种SRIR，输出一个预测的反射图。
	- Illumination Adjustment：调节亮度

- Experiment

	- Dataset：基于Cityscapes和Camvid生成的2458对图像，2118训练，340验证。100张真实低光照图像用作测试。
	- Train：每个子网络单独训练，分为四个阶段。
	- results：

- [code](https://github.com/XFW-go/ISSR)

- 思考

	- 没有用到perceptual loss，主要用到MSE、SSIM、Grad。
	- 文中只区分了天空陆地和前景，类别太少，对于真实场景增强，不够用

### [(IJCAI 2018)When Image Denoising Meets High-Level Vision Tasks A Deep Learning Approach](https://arxiv.org/pdf/1706.04284.pdf)

- 
- 第一个将去噪和语义分割及其他高级视觉任务结合的工作。
- Motivation

	- 由于许多算法用MSE做去噪，因此重要的细节很容易丢失，导致图像退化。
	- noise很容易影响网络的分类结果。

- Contribution

	- 提出一种连接图像去噪和高级视觉任务的方法，同时最小化图像重建损失和高级任务损失。证明了语义信息对去噪的重要性。
	- 证明了用共同损失训练网络，不仅可以增强去噪网络的效果，并且可以提高高级视觉任务的精确度。而且这样训练出来的去噪网络在各种高级视觉任务上都能够泛化。

- Method

	- 
	- 
	- Loss：
重建损失使用MSE损失。
高级视觉任务使用CE损失。

- Experiment

	- Dataset：
	- Train：高级视觉任务的网络使用一个在无噪声情况下训练好的网络，然后固定参数，用共同损失更新去噪网络的参数。

### [(CVPR 2018)Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform](https://arxiv.org/abs/1804.02815)

- 
- SFTGAN：一种超分领域GAN-based-method
- Motivation

	- 图像的纹理信息恢复的不好，通过改进损失函数有所改善，但还是存在问题。
	- perceptual loss和adversarial loss带来的细节和原图的语义关联度不够，不能反映真实的类别。实验说明，先验可以很大程度影响模型的output，因此语义先验是有作用的。
	- 为每个类别训练一个超分模型显然不可能。如果简单地把语义图和中间的feature map concat起来又没法充分发挥语义图的作用

- Contribution

	- 提出了一种Spatial feature transform模块(SFT)
节约参数。
即插即用，很容易与现有模型结合。
可扩展，prior不仅仅可以是语义图，也可以是深度图等等。 

- Method

	- 
	- condition network：把语义图转换为条件图，共享到所有的SFT块。
	- SFT块：通过条件图映射为一对参数，对特征进行变换。
	- Loss：
用预训练的VGG输出特征图计算perceptual loss，用minimax 计算对抗损失。

- Experiment

	- Dataset：预训练使用Imagenet，去掉30k以下的图片，取了450k训练图片。
fine-tune使用从搜索引擎收集的，训练10324，测试300。
	- Train：
	- results：数量分析，还有user study

- [code](https://github.com/xinntao/SFTGAN)

- 思考

	- 也是使用语义先验，类别有八个，但是还是不算多。没有提到语义图分割不对情况下的副作用。

### (CVPR 2021) GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution

- 
- Motivation
- Contribution
- Method
- Experiment

	- Dataset：
	- Train：

### [(ECCV 2020)Blind Face Restoration via Deep Multi-scale Component Dictionaries](https://arxiv.org/abs/2008.00418)

- 
- Motivation

	- blind image restoration对于恢复真实LQ图像有很强的实用价值
	- 缺陷：其中两种方法，1. 需要一个来自同样人物的正面HQ图像。2. HQ和LQ中姿势和表情不同会影响重建的效果

- Contribution

	- 使用一个五官部位字典作为参考备选，指导人脸重建。不需要对应的HQ图像
	- 提出一个DFT block，使用CAdaIN消除输入和字典的风格差异，使用confidence score控制融合程度。
	- 渐进式方法训练

- Method

	- 
	- Off-line 多尺度部位字典生成：VggNet作为特征提取器，在多个尺度上，先使用RoIAlign裁剪出四种部位，之后使用K-means分为多个堆，作为字典。
	- 采用同样的预训练VggNet作为encoder，保证同样的特征空间。
在DFT block中，首先通过RoIAlign，产生四个部位区域。假设input的部位和字典里的风格不一样，所以使用CAdaIN进行正则化，之后在Feature Match（用内积算相似度）在字典中选取相似的类。之后预测一个confidence score，在融合时更好的提供互补信息。最后Reverse RoIAlign把增强后的特征贴到相同位置。借用SFT块思想预测一对参数，输入到decoder。

		- CAdaIN
		- Confidence score

	- Loss：
reconstruction loss：包含MSE和perceptual loss
adversarial loss

- Experiment

	- Dataset：
FFHQ：10000张图像用来建立字典
VggFace2：训练
测试：2000VggFace2，2000CelebA
	- Train：

- [code](https://github.com/csxmli2016/dfdnet/)

- 思考

	- 不是完全的语义分割先验，只是将面部的一些部位进行裁剪并着重处理，整体结构比较复杂，需要先建立一个字典才能实现。
和有前脸参考的方法相比，本方法不需要成对的HQ图像，但是由于字典的限制，面部细节只能着重处理左右眼、鼻子、嘴巴这四个部位，还是有一定的提升空间

## 分支主题 5


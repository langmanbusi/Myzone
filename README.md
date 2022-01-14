# Papers and Resourse
## 经典
#### ✅｜[(2016 CVPR)Resnet](https://arxiv.org/pdf/1512.03385.pdf)
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- [code]


#### ✅｜[(NIPS 2017)Transformer](https://arxiv.org/abs/1706.03762) 
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko- reit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- [code]

#### ✅｜ViT

#### ✅｜GAN

#### ✅｜MoCo

## Review

#### ✅｜(2020 TPAMI)EventVisionSurvey


#### (2020 IEEESPM)Event-Based Neuromorphic Vision for Autonomous Driving: A Paradigm Shift for Bio-Inspired


#### (2020 TITS)NeuroIV_Neuromorphic Vision Intelligent Vehicle Safe Driving New Database Baseline Evaluations


#### (2021 CVPRW)Duwek_Image_Reconstruction_From_Neuromorphic_Event_Cameras_Using_Laplacian-Prediction_and_Poisson


#### (2021 Complexity)Data-Driven Technology in Event-Based Vision

## Event camera

#### ✅｜[(2018 CVPR)Event-based Vision meets Deep Learning on Steering Prediction for Self-driving Cars](https://arxiv.org/pdf/1804.01310.pdf)
- Ana I. Maqueda, Antonio Loquercio, Guillermo Gallego, Narciso Garc ́ıa, and Davide Scaramuzza
- [code]
- <image src="https://user-images.githubusercontent.com/89068081/149519450-fd180a0c-b8e2-4e00-8068-2fdfcd10b617.png" height="100">
- ![image](https://user-images.githubusercontent.com/89068081/149519505-e28a6bd6-0c67-4213-97aa-c71be91cc4f9.png)
- 传统相机的偏差较大，事件相机的预测精确。传统相机在高速行驶以及不同光照条件下，图像会模糊并产生伪影，事件相机时间分辨率高，动态范围大。
* 主要工作
  * 1)：首次实现了基于大规模事件相机数据的深度学习回归任务，并分析了为何事件相机数据能在这类任务上更有优势；
  * 2)：采用的深度卷积网络结构是现有的ResNet，并展示了基于ImageNet数据集的迁移学习能使训练得到更好的结果；
  * 3)：与现有转向预测的方法进行了对比，体现了本文结果的精确性


#### ✅｜[(2018 RSS)EV-FlowNet:Self-Supervised Optical Flow Estimation for Event-based Cameras](https://arxiv.org/pdf/1802.06898.pdf)
- Alex Zihao Zhu, Liangzhe Yuan, Kenneth Chaney and Kostas Daniilidis
- [code]
- ![image](https://user-images.githubusercontent.com/89068081/149522308-aee8637d-4d58-41be-8c38-4957ba4a1d6a.png)
- 首次运用FlowNet类似的下采样上采样结构进行光流估计。采用4CH输入数据，前两个通道是每个像素处发生的正事件和负事件的数量，后两个通道中的像素分别编码为该像素上最近的正事件和负事件的时间戳。输入大小为256×256×4的随机裁剪和旋转后的上述四通道图。绿色为下采样（编码）部分，通过步长为2的卷积实现，每一层的卷积结果保留，作为跳层链接到上采样（解码）层。在四层下采样（编码）后，中间两个蓝色的为两个残差块，对特征进行进一步提取。后面黄色的为上采样（解码）部分，通过对称padding实现。每一层的结果通过一个卷积核大小为1\*1的卷积，称为二通道的光流估计图（即图中向上的箭头），然后将在这个尺度计算一个loss，之后将这个光流估计图和这一层原来的图以及跳层链接的图进行cat连接，再通过上采样进入下一层。


#### ✅｜[(2019 ICCV)End-to-End Learning of Representations for Asynchronous Event-Based Data](http://rpg.ifi.uzh.ch/docs/ICCV19_Gehrig.pdf)
- Daniel Gehrig, Antonio Loquercio, Konstantinos G. Derpanis, Davide Scaramuzza
- [code](https://github.com/uzh-rpg/rpg_event_representation_learning)
- ![image](https://user-images.githubusercontent.com/89068081/149520032-b7ef3250-b317-4026-9bca-9bc3df862ad7.png)
- EST事件数据表示方法的优势在于，对于第一章中提到的四元组事件的位置、时间、极性信息可以全部利用上，因此可以充分利用事件流的高刷新率、高动态范围等优势。EST的建立过程主要有三步，在第一步中，作者引入连续的测量函数来具体化每个事件，将脉冲信号转化为一个具体的量，使其具有意义，第二步需要让事件序列继续与一个核变换函数进行卷积操作，此举可以使事件数据表现出更多可学习的特征，最后对得到的连续事件数据进行取样，使其离散化，得到最终的EST模型
- ![image](https://user-images.githubusercontent.com/89068081/149520012-b0f383bf-4a6a-4677-a9cb-bbff6a217cf0.png)
- 本文的另一个贡献在于实现了一种端到端的事件表示方法。对于核函数，作者直接引入了MLP作为卷积核，让网络自己决定需要什么样的核来提取事件数据的特征。因此在通过测量函数实值化时间序列之后，就进入了第一个可训练的网络，之后再离散化得到EST模型，以传统图像意义下的多通道的帧形式直接输入深度神经网络，最终输出结果，计算损失函数，同时反向传播到处理过程中的MLP和特征提取的ResNet，进行参数更新


#### (2019 WACV)Space-Time_Event_Clouds_for_Gesture_Recognition_From_RGB_Cameras_to_Event_Cameras
- [code]


#### ✅｜[(CVPR 2020)Recycling video dataset for event cameras](http://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf) 
- Daniel Gehrig, Mathias Gehrig, Javier Hidalgo-Carrio ́, Davide Scaramuzza
- [code](https://github.com/uzh-rpg/rpg_vid2e)
- ![image](https://user-images.githubusercontent.com/89068081/149520771-251b24d1-9238-4e44-b7b1-791e4d328a52.png)
- 提出了一个把现有的视频数据转换成事件数据的框架，使更多应用成为现实。在这些合成事件数据上训练的模型，在真实事件数据上有很好的泛化性，即使是在极端场景下。在识别和语义分割任务中评估，文中的方法可以用来作为一种微调提高性能。
- 方法：
  - 第一步利用最新的帧插值技术17，使用自适应上采样技术将低帧速率视频转换为高帧速率视频。
  - 第二步利用ESIM从视频产生出事件，算法在32的3.1。为了促进domain adaptation 提出了两种技术，均匀分布随机化正负阈值。
  - 最后用文献15中的方法将稀疏异步的事件转化为张量表示。


#### ✅｜[(CoRL 2018)ESIM-an Open Event Camera Simulator](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf)
- [code](https://github.com/uzh-rpg/rpg_esim)


#### [(ITSC 2020)DDD20 End-to-End Event Camera Driving Dataset: Fusing Frames and Events with Deep Learning for Improved Steering Prediction](https://arxiv.org/pdf/2005.08605.pdf)
- Yuhuang Hu, Jonathan Binas, Daniel Neil, Shih-Chii Liu and Tobi Delbruck
- [code]
- Using DDD20, we report the first study of fusing brightness change events and intensity frame data using a deep learning approach to predict the instantaneous human steering wheel angle


#### [(CVPR 2020)EV-SegNet: Semantic Segmentation for Event-based Cameras](https://drive.google.com/file/d/1eTX6GXy5qP9I4PWdD4MkRRbEtfg65XCr/view)
- Inigo Alonso, Ana C. Murillo
- [code](https://github.com/xupinjie/Ev-SegNet)


#### [(CVPR 2021)Learning to Reconstruct High Speed and High Dynamic Range Videos from Events](https://openaccess.thecvf.com/content/CVPR2021/papers/Zou_Learning_To_Reconstruct_High_Speed_and_High_Dynamic_Range_Videos_CVPR_2021_paper.pdf)
- Yunhao Zou Yinqiang Zheng Tsuyoshi Takatani Ying Fu 
- [code]()

#### [(CVPR 2020)(TPAMI 2021)E2SRI-Learning to Super Resolve Intensity Images from Events](http://openaccess.thecvf.com/content_CVPR_2020/papers/I._Learning_to_Super_Resolve_Intensity_Images_From_Events_CVPR_2020_paper.pdf)
- Mohammad Mostafavi, Jonghyun Choi and Kuk-Jin Yoon
- [code](https://github.com/gistvision/e2sri)
- ![image](https://user-images.githubusercontent.com/89068081/149351304-c9a836e2-cf51-4f03-a991-7f7beea0f11e.png)


#### [(ECCV 2020)Event Enhanced High-Quality Image Recovery](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580154.pdf)
- Wang, Bishan and He, Jingwei and Yu, Lei and Xia, Gui-Song and Yang, Wen
- [code](https://github.com/ShinyWang33/eSL-Net)


## Multi-modal fusion (Event camera)

#### ✅｜[(IEEE RAL 2021)Combining events and frames using recurrent asynchronous multimodal network for monocular depth estimation](http://rpg.ifi.uzh.ch/RAMNet.html) 
- Daniel Gehrig, Michelle Ru ̈egg, Mathias Gehrig, Javier Hidalgo-Carrio, Davide Scaramuzza
- [code](http://rpg.ifi.uzh.ch/RAMNet.html)
- ![image](https://user-images.githubusercontent.com/89068081/149520993-ff0bbce3-7eb6-41b1-aed6-4615137fe6b9.png)
- ![image](https://user-images.githubusercontent.com/89068081/149521160-b0bce242-c6a6-43ba-b625-d44eb566de36.png)
- Related work: SLAM:12, featuretracking:13,14,15, HDR reconstruction:16, deblurring:17, monocular depth:6,19,20,(7,21)(循环结构), fusion:9,10,22
                RNN struggle:4,5,23,24(padding,copying,sampling), neural ODE:5, phased LSTM:4, 


#### ✅｜[(ICIAP 2019)Video synthesis from Intensity and Event Frames](https://iris.unimore.it/retrieve/handle/11380/1178955/233862/ICIAP19_Event_Cameras.pdf)
- Stefano Pini, Guido Borghi, Roberto Vezzani, Rita Cucchiara
- [code]
- ![image](https://user-images.githubusercontent.com/89068081/149521325-a1633963-0fc6-4544-8406-a98eab996bbf.png)
- 一个全卷积encoder-decoder结构，将frame和event frame concat之后，预测将来的frame。


#### ✅｜[(ICMVA 2021)Standard and Event Cameras Fusion for Feature Tracking](https://dl.acm.org/doi/10.1145/3459066.3459075)
- Yan Dong, Tao Zhang
- [code](https://github.com/LarryDong/FusionTracking)


#### [(CVPR 2021)EvDistill](https://arxiv.org/pdf/2111.12341.pdf)
- Lin Wang, Yujeong Chae, Sung-Hoon Yoon, Tae-Kyun Kim, Kuk-Jin Yoon
- [code](https://github.com/addisonwang2013/evdistill)

#### [(CVPRW 2021)EFI-Net_Video_Frame_Interpolation_From_Fusion_of_Events_and_Frames_CVPRW_2021_paper](https://tub-rip.github.io/eventvision2021/papers/2021CVPRW_EFI-Net_Video_Frame_Interpolation_from_Fusion_of_Events_and_Frames.pdf)
- Genady Paikin, Yotam Ater, Roy Shaul, Evgeny Soloveichik
- [code]()

## Transformer multi-modal fusion (Event camera)

#### ✅｜[(ICCV 2021)Event-Based_Video_Reconstruction_Using_Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Weng_Event-Based_Video_Reconstruction_Using_Transformer_ICCV_2021_paper)
- Wenming Weng, Yueyi Zhang, Zhiwei Xiong
- [code](https://github.com/WarranWeng/ET-Net)
- ![4D297A5B81BA06AF6422557D3B1E25ED](https://user-images.githubusercontent.com/89068081/149521796-3ad2e373-bb14-44d1-aed2-0c44ffd079d4.png)
- 提出一种CNN和Transformer的杂交网络，分为RCB，TPA和MLU三个部分，RCB进行下采样，获得多尺度，以及利用LSTM得到长期依赖关系，之后多个尺度输入到transformer中，通过编码器解码器提取多尺度上下文。MLU将TPA得到的全局上下文特征和RCB得到的局部特征结合并上采样，最后经过一个tail得到最终的复原图片。

## Transformer multi-modal fusion (Other sensors)

#### ✅｜[(CVPR 2021)Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/pdf/2104.09224.pdf) 
- Aditya Prakash, Kashyap Chitta, Andreas Geiger
- [code](https://github.com/autonomousvision/transfuser)
- ![image](https://user-images.githubusercontent.com/89068081/149521851-e6a3c728-e0ce-4c8c-a904-11d290770057.png)
- LiDAR和frame的融合，使用transformer。利用resnet结构获得不同尺度的特征图，而transformer则在相同尺度内融合两种模态的数据和全局信息


## Utils

#### ✅｜[(CVPR 2018)Super SloMo-High Quality Estimation of Multiple Intermediate Frames for Video Interpolation](https://arxiv.org/abs/1712.00080) 
- [code](https://github.com/avinashpaliwal/Super-SloMo)
- ![image](https://user-images.githubusercontent.com/89068081/149522158-d95e082e-7cb3-4b8f-8d63-a3f1617f76d6.png)


## Pretrain

#### [(AAAI 2022)Simple 2D Image and 3D Point Cloud Unsupervised Pre-training for spatial-aware visual representation](https://arxiv.org/abs/2112.04680) 
- Zhenyu Li, Zehui Chen, Ang Li, Liangji Fang, Qinhong Jiang, Xianming Liu, Junjun Jiang, Bolei Zhou, Hang Zhao
- [code](https://github.com/zhyever/SimIPU)
- pre-training for RGB and Event Camera; pre-train a model on large-scale pointcloud/event camera/Radar (any other sensors you can think of), and try to propose our own ideas on utilizing them for knowledge transfer


#### [(arxiv)PointCLIP: Point Cloud Understanding by CLIP](https://arxiv.org/abs/2112.02413) 
- Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, Hongsheng Li
- [code](https://github.com/ZrrSkywalker/PointCLIP)


## Dataset(including Real and Synthetic)

#### [MVSEC(R)](https://daniilidis-group.github.io/mvsec/download/)

#### DDD17(R)
- End-to-end DAVIS Driving Dataset, the first open dataset of annotated DAVIS driving recordings. The dataset contains more than 12 hours of recordings captured with a DAVIS sensor. Each recording includes both event data and grayscale frames along with vehicle information (e.g. vehicle speed, throttle, brake, steering angle). Recordings are captured in cities and highways, in dry and wet weather conditions, during day, evening, and night.

#### DDD20(R)

#### N-Caltech101(R)

#### [N-ImageNet(R)](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_N-ImageNet_Towards_Robust_Fine-Grained_Object_Recognition_With_Event_Cameras_ICCV_2021_paper.pdf)

#### HQF(R)
- The HQF dataset, recorded by two DAVIS240C [5] cameras, provides high quality ground- truth frames, of which the motion blur is maximally mit- igated under preferable exposure. 14 sequences are con- tained , covering a wider range of motions and scene types, including static scenes and motion scenes of slow, medium and fast, indoor and outdoor scenes.

#### IJRR(R)
- IJRR provides 25 real- istic datasets by DAVIS240C [5] and two synthetic datasets via the event camera simulator

#### Eventscape(S)

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


#### [(2018 RSS)EV-FlowNet:Self-Supervised Optical Flow Estimation for Event-based Cameras](https://arxiv.org/pdf/1802.06898.pdf)
- [code]


#### ✅｜[(2019 ICCV)End-to-End Learning of Representations for Asynchronous Event-Based Data](http://rpg.ifi.uzh.ch/docs/ICCV19_Gehrig.pdf)
- Daniel Gehrig, Antonio Loquercio, Konstantinos G. Derpanis, Davide Scaramuzza
- [code](https://github.com/uzh-rpg/rpg_event_representation_learning)


#### (2019 WACV)Space-Time_Event_Clouds_for_Gesture_Recognition_From_RGB_Cameras_to_Event_Cameras
- [code]


#### ✅｜[(CVPR 2020)Recycling video dataset for event cameras](http://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf) 
- Daniel Gehrig, Mathias Gehrig, Javier Hidalgo-Carrio ́, Davide Scaramuzza
- [code](https://github.com/uzh-rpg/rpg_vid2e)


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

#### [(ECCV 2020)Event Enhanced High-Quality Image Recovery](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580154.pdf)
- Wang, Bishan and He, Jingwei and Yu, Lei and Xia, Gui-Song and Yang, Wen
- [code](https://github.com/ShinyWang33/eSL-Net)


## Multi-modal fusion (Event camera)

#### ✅｜[(IEEE RAL 2021)Combining events and frames using recurrent asynchronous multimodal network for monocular depth estimation](http://rpg.ifi.uzh.ch/RAMNet.html) 
- Daniel Gehrig, Michelle Ru ̈egg, Mathias Gehrig, Javier Hidalgo-Carrio, Davide Scaramuzza
- [code](http://rpg.ifi.uzh.ch/RAMNet.html)
- Related work: SLAM:12, featuretracking:13,14,15, HDR reconstruction:16, deblurring:17, monocular depth:6,19,20,(7,21)(循环结构), fusion:9,10,22
                RNN struggle:4,5,23,24(padding,copying,sampling), neural ODE:5, phased LSTM:4, 


#### ✅｜[(ICIAP 2019)Video synthesis from Intensity and Event Frames](https://iris.unimore.it/retrieve/handle/11380/1178955/233862/ICIAP19_Event_Cameras.pdf)
- Stefano Pini, Guido Borghi, Roberto Vezzani, Rita Cucchiara
- [code]
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
- 提出一种CNN和Transformer的杂交网络，分为RCB，TPA和MLU三个部分，RCB进行下采样，获得多尺度，以及利用LSTM得到长期依赖关系，之后多个尺度输入到transformer中，通过编码器解码器提取多尺度上下文。MLU将TPA得到的全局上下文特征和RCB得到的局部特征结合并上采样，最后经过一个tail得到最终的复原图片。

## Transformer multi-modal fusion (Other sensors)

#### ✅｜[(CVPR 2021)Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/pdf/2104.09224.pdf) 
- Aditya Prakash, Kashyap Chitta, Andreas Geiger
- [code](https://github.com/autonomousvision/transfuser)
- LiDAR和frame的融合，使用transformer。利用resnet结构获得不同尺度的特征图，而transformer则在相同尺度内融合两种模态的数据和全局信息


## Utils

#### ✅｜[(CVPR 2018)Super SloMo-High Quality Estimation of Multiple Intermediate Frames for Video Interpolation](https://arxiv.org/abs/1712.00080) 
- [code](https://github.com/avinashpaliwal/Super-SloMo)

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

#### HQF
- The HQF dataset, recorded by two DAVIS240C [5] cameras, provides high quality ground- truth frames, of which the motion blur is maximally mit- igated under preferable exposure. 14 sequences are con- tained , covering a wider range of motions and scene types, including static scenes and motion scenes of slow, medium and fast, indoor and outdoor scenes.

#### IJRR
- IJRR provides 25 real- istic datasets by DAVIS240C [5] and two synthetic datasets via the event camera simulator

#### Eventscape(S)

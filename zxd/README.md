# 安装环境
ubuntu 18  
cuda 9  
python 3.6  
torch 1.0.0  
torchvision 0.2.0 
## Error
1. cocoapi安装出现`fatal error: Python.h: No such file or directory`
    - 当前python版本没有安装完全，执行
    ```sh
    sudo apt-get install python3-dev
    ```
    如果报错，那么就换一个python版本建虚拟环境吧，比如3.6或3.7
    
2. apex安装出现 `error: expected primary-expression before 'some' token`
    - 恢复到apex的早期版本
    ```sh
    git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
    ```
    
3. `cannot import name '_C' from 'maskrcnn_benchmark`
    - 主目录下的setup.py没有编译成功，重新编译
    ```python
    python setup.py build develop
    ```
    
4. `RuntimeError: _th_or not supported on CUDAType for Bool`
    - Just replace all `torch.bool` with `torch.uint8` in file `modeling/balanced_positive_negative_sampler.py` and file `structures/segmentation_mask.py`
# 参考
https://www.cnblogs.com/wangyong/p/10614898.html

简书： https://www.jianshu.com/p/ccf8affcdb98

硕士毕业论文
# 代码
cfg的参数由`config/defaults.py`和`--config-file`传入的yaml文件决定
## 数据
- 利用torchvision.datasets.coco.CocoDetection基类读取coco数据，读取的target的box一开始是(x,y,w,h)格式，后通过BoxList类转为（xyxy）,以像素值而非百分比的格式输入模型
- 数据增强的resize

  设w,h为原img大小，max,min为设定的期望大小，目的是把w,h保持比例进行resize。先看看以min为基准resize后最大值是否>max，如果超过了max,就对min进行调整（减小），然后再以min为基准进行resize，完毕。

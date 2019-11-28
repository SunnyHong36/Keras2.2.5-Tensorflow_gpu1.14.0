# Keras2.2.5-Tensorflow_gpu1.14.0
Deep Learning with GPU Accelerate under Window10 
# Tensorflow-gpu 的安装方法及步骤
　　首先文章写于2019年11月28日，在折腾了两天tensorflow-gpu安装后终于成功。由于前段时间清华的anaconda源停止了镜像服务，再加上6月份国内的网络环境大家都懂，那个时候是第一次想着安装GPU版的Tensorflow(后简称TF)加速，由于这些限制，当时就放弃了. 前两天折腾一个项目，想着还得搞一下Keras深度学习，一看Tensorflow都更新到了2.0版本了，网上铺天盖地的说2.0版的就是个Keras,也尝试装了一下2.0版本的，CUDA和cuDNN也对应上了，导入Keras函数时不太好使，查看了TF的github的文档具体网址：https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md ，好像函数啥的都变了，就卸载了，重新安装了较低版本的TF(国内的网站例如CSDN上的教程都是装1.5，1.8这些版本，有点太老了)1.14.0，因为我没仔细看1.15的更新说明，只知道1.15是最后一个1.x的版本(凌晨2点多躺在床上看的),所以就尝试了1.14，毕竟TF的版本变化太快，对于我这种只是拿它当Keras后端用于快速建模的来说，也没必要使用最新版本，而且现在能找到的关于Keras教程都没有用很新的tensorflow后端，贸然装了最新版可以预料后面学习时肯定各种库方法找不到，况且更新文档里说了那么多函数变化，目前没工夫去研究，所以就凑合这个版本用。而且以目前的趋势，pytorch也在逐渐火起来，可以初步学习完TF再学学小火炬。下面说明一下安装过程，给自己做个笔记，也填一下国内网上搜到的安装教程的坑。
## 1.首先安装Anaconda创建一个专门给这个项目的虚拟环境
　　这个Anaconda的安装就不需要多做解释，然后换源成国内的清华源。创建新的虚拟环境，并提前装一下pandas,numpy,matplotlib,scipy等库，不装也行，conda会自己解决这些依赖，最坏的情况就是提醒你，到时候再装也可以。这个教程在网上能找到很多。

几个命令：
```
conda create -n tfenv(这个是给要创建的虚拟环境的名字) python=3.5(指定python版本)  #创建虚拟环境
conda env list #查看有哪些环境
conda info --env #也是查看环境
conda activate tfenv  #激活环境
conda deactivate  #退出环境
```
## 2.在虚拟环境中安装tensorflow-gpu指定版本
主要是命令:
```
#这个-i参数是指临时使用后面这个清华的源来安装我要装的package,鉴于国内的网络环境这样安装pkg(package)是最好的
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.14.0
```
　　使用pip安装不使用conda install, 因为conda install会给你安装别的包。我后面尝试pip装好了tensorflow-gpu后用的conda install keras，就又给我装tensorflow-gpu，查看conda list的时候两个都有，一个是用pypi来build，一个是什么pyxx(忘了具体)build的。

## 3.准备GPU驱动（显卡驱动）、CUDA和cuDNN
　　网上的教程写的真的是乱七八糟，据我看完了20几篇加上自己的实践总结出来的这个是最全的，能把好多没讲明白的问题都涉及到。  
　　首先我们可以查看TF的官网 https://www.tensorflow.org/ ， 这个国内网络环境可以打开,真不行可以百度搜索tensorflow,有一个是 https://tensorflow.google.cn/ , 还有它的中文社区，都可以。官网中有个地方讲的就有点自相矛盾，在安装的这一页的左边，先上图：　　
<p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-3b8e22206a210a9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width = 20% height = 20% /> 
 <p align="center">
  <em>官网安装页左边</em>
  </p>  
</p>

<br>找到**Build from source**然后选择**Windows**：<br><p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-e8ae6f6ddd82bc79.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" /> 
 <p align="center">
  <em>Build from source</em>
  </p>  
</p> <br>进去之后拉到页面最下方查看官方编译过的GPU版本的TF所需要的支持

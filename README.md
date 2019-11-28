# Keras2.2.5-Tensorflow_gpu1.14.0
Deep Learning with GPU Accelerate
# Tensorflow-gpu 的安装方法及步骤
首先文章写于2019年11月28日，在折腾了两天tensorflow-gpu安装后终于成功。由于前段时间清华的anaconda源停止了镜像服务，再加上6月份国内的网络环境大家都懂，那个时候是第一次想着安装GPU版的Tensorflow(后简称TF)加速，由于这些限制，当时就放弃了. 前两天折腾一个项目，想着还得搞一下Keras深度学习，一看Tensorflow都更新到了2.0版本了，网上铺天盖地的说2.0版的就是个Keras,也尝试装了一下2.0版本的，CUDA和cuDNN也对应上了，导入Keras函数时不太好使，查看了TF的github的更新文档，好像函数啥的都变了，就卸载了，重新安装了较低版本的TF(国内的网站例如CSDN上的教程都是装1.5，1.8这些版本，有点太老了)1.14.0，因为我没仔细看1.15的更新说明(毕竟是在今天凌晨2点多躺在床上看的),所以就尝试了1.14，想着也够用。下面说明一下安装过程，给自己做个笔记，也填一下国内安装教程的坑。
## 1.首先安装Anaconda创建一个专门给这个项目的虚拟环境
这个Anaconda的安装就不需要多做解释。创建新的虚拟环境，并提前装一下pandas,numpy,matplotlib,scipy等库，不装也行，conda会自己解决这些依赖，最坏的情况就是提醒你，到时候再装也可以。
## 2.在虚拟环境中安装tensorflow-gpu指定版本
主要是命令:
```pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.14.0```
使用pip安装不使用conda install, 因为conda install会给你安装别的

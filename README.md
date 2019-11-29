# Keras2.2.5-Tensorflow_gpu1.14.0
Using Keras to Practice Deep Learning with GPU Accelerate under Window10 
# Tensorflow-gpu 的安装方法及步骤
　　首先文章写于2019年11月28日，在折腾了两天tensorflow-gpu安装后终于成功。由于前段时间(距离写这个文章应该是很前一段时间了，嘻嘻)清华的anaconda源停止了镜像服务，再加上6月份国内的网络环境大家都懂，那个时候是第一次想着安装GPU版的Tensorflow(后简称TF)加速，由于这些限制，当时就放弃了。 前两天折腾一个项目，想着还得搞一下Keras深度学习，一看Tensorflow都更新到了2.0版本了，网上铺天盖地的说2.0版的就是个Keras,也尝试装了一下2.0版本的，CUDA和cuDNN也对应上了，导入Keras函数时不太好使，查看了TF的github的文档具体网址：https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md ，好像函数啥的都变了，就卸载了，重新安装了较低版本的TF(国内的网站例如CSDN上的教程都是装1.5，1.8这些版本，有点太老了)1.14.0，因为我没仔细看1.15的更新说明，只知道1.15是最后一个1.x的版本(凌晨2点多躺在床上看的),所以就尝试了1.14，毕竟TF的版本变化太快，对于我这种只是拿它当Keras后端用于快速建模的来说，也没必要使用最新版本，而且现在能找到的关于Keras教程都没有用很新的tensorflow后端，贸然装了最新版可以预料后面学习时肯定各种库方法找不到，况且更新文档里说了那么多函数变化，目前没工夫去研究，所以就凑合这个版本用。而且以目前的趋势，pytorch也在逐渐火起来，可以初步学习完TF再学学小火炬。下面说明一下安装过程，给自己做个笔记，也填一下国内网上搜到的安装教程的坑。
## 1.首先安装Anaconda创建一个专门给这个项目的虚拟环境
　　这个Anaconda的安装就不需要多做解释，然后换源成国内的清华源。创建新的虚拟环境，并提前装一下pandas,numpy,matplotlib,scipy等库，不装也行，conda会自己解决这些依赖，最坏的情况就是提醒你，到时候再装也可以。这个教程在网上能找到很多。

几个命令：
```
conda create -n tfenv python=3.5  #创建虚拟环境   
#tfen--这个是自己给要创建的虚拟环境起的名字      
#python=3.5--是指定该虚拟环境里的安装3.5版本的python
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
　　如果不知道版本号，可以瞎写一个，然后会提示你有哪些版本号，再重新写命令，下载正确的版本。
  <br>这里使用pip安装不使用conda install, 因为conda install会给你安装别的包。我后面尝试pip装好了tensorflow-gpu后用的conda install keras，就又给我装tensorflow-gpu，查看conda list的时候两个都有，一个是用pypi来build，一个是什么pyxx(忘了具体)build的。

## 3.准备GPU驱动（显卡驱动）、CUDA和cuDNN
　　网上的教程写的真的是乱七八糟，据我看完了20几篇教程加上自己的实践总结出来的这个可以说是目前最全的，把CUDA，cuDNN,显卡驱动这些问题都讲了一下。通过查看网上这些资料我们知道了要安装GPU版本的TF需要**具备的条件**是：
 <br>1.**硬件**：一块NVIDIA®的GPU卡，也就是俗称的N卡，这块卡的CUDA®计算能力要不低于3.5，一般的N卡笔记本电脑都是可以的，要是不知道的话，可以根据自己显卡型号在这个网站 https://developer.nvidia.com/cuda-gpus 上查一下，打开这个网站应该是需要网速和耐心等待的(查看自己电脑显卡型号的方法：打开NVIDIA控制面板，点击左下角系统信息，“显示”里就是，如下图所示)。 <br><p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-b9dc598452d590b5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>查看显卡型号</em>
  </p>  
</p>
查看完了之后根据英伟达官网上的指示选择自己显卡类型去查看算力值:

<br><p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-93a557f4f3bdaa23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=80% /> 
 <p align="center">
  <em>选择显卡类型</em>  
  </p>  
</p>

<br><p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-1cd1f4da733e2d6d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=90% /> 
 <p align="center">
  <em>查看显卡算力</em>   
  </p>  
</p>

<br>2.**软件**:显卡驱动、需要根据安装的TF版本选择相对应的CUDA和cuDNN，总共三个软件，显卡驱动不用多说，都能下载到，不会的可以自行百度。剩下的两个软件分别是在https://developer.nvidia.com/cuda-toolkit-archive 和https://developer.nvidia.com/rdp/cudnn-archive 这两个地方下载，至于选择下载哪个接下来说。
<br>我们可以查看TF的官网 https://www.tensorflow.org/ ， 这个国内网络环境可以打开,真不行可以百度搜索tensorflow,有一个是 https://tensorflow.google.cn/ , 还有它的中文社区，都可以。官网中有个地方讲的就有点自相矛盾，在安装的这一页的左边，先上图：　　
<p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-3b8e22206a210a9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width = 25% height = 25% /> 
 <p align="center">
  <em>官网“安装”页左边</em>
  </p>  
</p>

<br>找到 **Build from source** ,然后选择 **Windows** ：
<br><p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-e8ae6f6ddd82bc79.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" /> 
 <p align="center">
  <em>Build from source</em>
  </p>  
</p> 

<br>进去之后拉到页面最下方查看官方编译过的GPU版本的TF所需要的支持 

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-b3c419f4dbd3ad61.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=80% /> 
 <p align="center">
  <em>GPU版TF对应的CUDA和cuDNN版本</em>
  </p>  
</p>

这里1.13.0版本官网给出的是CUDA9和cuDNN7。但是同样的在官网安装页左边，找到**Additional setup**进入**GPU support**里面是另外一种说法：
<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-e5dea41a133e1587.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>官网“安装”页左边“GPU support”入口</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-0d470bb91e4b3c3b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>官网对软件的要求</em>
  </p>  
</p>
<br>这里说的软件需求是:<br>　　·GPU drivers--显卡驱动，而且后面如果要装CUDA10.0的话需要显卡驱动的版本在410.x或者更高，这个就在上面说的NVIDIA控制面板-->左下角系统信息-->显示-->驱动程序版本里，我的是441.41，完全满足了。
<br>　　·CUDA Toolkit--这个是我们安装CUDA的时候会自动安装的，后面这句就比较重要了，说的是当TF版本在1.13.0以上时支持的是CUDA10.0，我当时按照第一个地方的信息装的是CUDA9，就没有成功。所以这里注意，**官网有些说明不太准确，准确信息还是要到Tensorflow项目的github上去查看**，哪个版本对应哪个CUDA都写的比较明白。
<br>　　·CUPTI--这个也是在装CUDA的时候就会装上
<br>　　·cuDNN SDK(>=7.4.1)--也就是cuDNN这个软件要是7.4.1版本以上
<br>　　·TensorRT 5.0--这个是可选项，暂时可以不装

<br>然后由于我装的是1.14.0版本的TF，所以这里选择下载CUDA10.0版本，在[CUDA的下载网页](https://developer.nvidia.com/cuda-toolkit-archive)中选择：

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-b00fd3474e64627a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=90% /> 
 <p align="center">
  <em>CUDA版本选择</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-164856ca6330a952.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>根据电脑系统选择</em>
  </p>  
</p>

<br>　　这里选择哪个10.0版本的，点进去后根据自己的电脑系统选择，local版是下载到本地的一个安装包，network版是联网安装，这个网站本来国内就很难打开，再联网安装就不用想，必定失败！
<br>　　进入[cuDNN的下载地址](https://developer.nvidia.com/rdp/cudnn-archive)后,发现支持CUDA10.0的cuDNN已经更新到了7.6.4版本了，没敢用最新版的，就随便挑了个7.6.0版本的下载了。

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-083217c4bc87b2b1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=90% /> 
 <p align="center">
  <em>选择喜爱的cuDNN版本</em>
  </p>  
</p>

<br>下载完毕后，**建议先升级显卡驱动**，不升级的话后面说怎么操作。开始安装CUDA，双击后按照默认的下一步，选择自定义安装，有三个选项“CUDA”，“Other components”和“Driver components”,其中“CUDA”选项是全选，这里有一项是关于安装Visual Studio的，可以不选，但是因为我电脑上已经装了Visual Studio,所以全选也没关系，当然如果电脑没装VS的话，选了也没事，后面会有个提醒，忽视就好了。你要是不放心的话就提前装一个Visual Studio,只需要2015版的就可。第二项这里点开加号后如果是“当前版本”比“新版本”高就不要选，反之就可以选上，第三项也是，第三项是因为CUDA这个软件**自带了一个NVIDIA的显卡驱动**，如果我们在装CUDA之前升级了显卡驱动，这个时候一般是当前版本比新版本要高的，而且升级了显卡驱动后也是可以看到自己的显卡目前能支持到CUDA的哪个版本，在NVIDIA控制面板-->系统信息-->组件-->3D设置-->NVCUDA里查看，我这个升完显卡驱动，已经支持到了10.2.95了，所以说我给我的TF使用10.0的CUDA我的显卡是完全可以胜任这个工作的。这个CUDA是显卡驱动用的CUDA，跟我们这里TF要用的CUDA没关系，我们电脑里可以安装很多不同版本的CUDA，但是得按照教程的方法设置，才能跟对应版本的TF结合使用。自定义选好要安装的东西之后继续下一步，它的默认安装位置在C盘，当然想改的话就要**记住你的安装位置**（这个我没试过装别的盘是否好用）。


<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-24aff5124b751489.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>CUDA安装过程--检查系统兼容性，一般没问题</em>
  </p>  
</p>


<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-f76641a744a371bb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>CUDA安装选项选择自定义</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-4185b164f1a017de.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>CUDA自定义选项里有哪些内容</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-6d636379ca352f0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>CUDA安装位置，这里是我装9.2版本的截图，实际的应该是把这里的9.2都换成10.0</em>
  </p>  
</p>

<br>接下来是cuDNN的安装，把下载下来的cuDNN解压缩后，里面的三个文件夹复制，然后到CUDA的安装目录下，我这里默认的安装路径应该是上图中的安装位置的第一个地方，如果是改过的话，你刚才记住的安装位置就有用了。用cuDNN里的这三个文件夹覆盖CUDA里原有的三个同名文件夹就完成了安装。

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-9d142b153e3a8258.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>cuDNN压缩包解压</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-30ddaacd63133b6d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>CUDA的安装位置</em>
  </p>  
</p>

<br>现在已经完成了绝大部分的TF安装工作了，距离使用TF就一步了。接下来是把几个路径添加到环境变量，这个也比较简单：此电脑-->右键属性-->高级系统设置-->高级-->环境变量-->系统变量-->Path-->编辑-->新建，然后把图中这前四个路径添加就可完成所有TF安装工作，接下来就可以使用TF了（**普天同庆啊，有木有！**）。

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-60ae5f83a2778374.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>此电脑-->右键属性-->高级系统设置</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-a1a00a4dddb38fd6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>高级-->环境变量</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-87ff3a7af3a36832.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>系统变量-->Path-->新建</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-5de21c792d2ddd42.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>添加前四个环境变量</em>
  </p>  
</p>

<br>这个时候可以**查看是否已经使用GPU**：

命令是
```
#进入环境
conda  activate tfenv
#执行python
python
#导入tensorflow查看
>>>import tensorflow as tf
>>>tf.test.gpu_device_name()
```
会显示device是GPU

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-d198856e6929c9cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>查看是否使用GPU</em>
  </p>  
</p>

<br>**补充说明如果装错了CUDA版本,怎么卸载重装**:打开控制面板,选择程序,卸载程序,把NVIDIA的除了图中后面三项,即:NVIDIA PhysX 系统软件9.19.0218,NVIDIA GeForce Experience 3.20.1.57和NVIDIA 图形驱动程序441.41,其余全部删掉,其中删除有的项需要重启电脑才能生效。

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-dc3341d41a317364.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>控制面板</em>
  </p>  
</p>

<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-16d37ff057c11f40.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>需要卸载的CUDA项</em>
  </p>  
</p>
<br>删除完CUDA的这些项后，查看CUDA安装位置，应该是有三个文件夹还在，因为这三个文件夹是我们从cuDNN里复制粘贴过来的，所以手动删除；最好是再清理一下注册表，不会自己清注册表的话就用 [火绒](https://www.huorong.cn/) 的安全工具清理,要是你装了360，腾讯电脑管家，金山什么的，就不多说啥了，卸载你的这些软件，然后装个火绒吧，不过看目前火绒的趋势有向360靠拢的倾向，如果它不行了，我就会删除这部分推荐的。然后再查看一下环境变量，有可能跟你装错的这个版本有关的环境变量还在，手动删除就可。这样就完成了删除，然后重新安装需要的版本。

# Keras安装
　　安装指定的2.2.5版本的keras,还是使用pip来安装：
 <br>命令是
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.2.5
#pip install --ignored-installed --upgrade pkg  
# --ignored-installed   这个参数忽略已安装的
#--upgrade    一般搭配前面的参数一起用来强制升级某个包，也可能没有这个参数，因为忘记是否装了某个包
```
**查看Keras是否可用**：
激活环境，打开python
```
import keras
```
会告诉我们使用TF做后端
<br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-a57d669b217a95de.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=100% /> 
 <p align="center">
  <em>导入keras</em>
  </p>  
</p>

现在就可以使用keras来实战深度学习了！

# Jupyter notebook中使用安装好的以Tensorflow-GPU为后端的Keras环境
要实现这个事情有三个方法：
  <br>1.最简单的方法：在base环境里安装插件，命令是
  ```
  conda install nv_conda
  #进入要使用的环境
  conda activate tfenv    #tfenv换成自己的环境名字
  conda install -y jupyter
  ```
  <br>2.给每个环境都安装ipykernel的包，命令是
  ```
  conda install -n tfenv ipykernel
  #进入环境安装
  conda activate tfenv
  python -m ipykernel install --user
  ```
  <br>3.在创建环境的时候就预装ipykernel的包，命令是
  ```
  conda create -n tfenv python=3.5 ipykernel
  ```
  
  <br>**打开jupyter notebook,选择要用的环境，新建一个文件，测试**：
  结果如图所示：
  
  <br> <p align="center">
<img src="https://upload-images.jianshu.io/upload_images/20306957-89f5782b4ff8af00.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width=150% /> 
 <p align="center">
  <em>Jupyter notebook测试导入keras</em>
  </p>  
</p>

至此就可以在Jupyter notebook里为所欲为之为所欲为了。

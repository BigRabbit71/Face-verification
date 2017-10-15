# Face-verification

> * **What's the task?**
> * **What's the dataset?**
> * **Which Scheme to choose?**
> * **How to use the code?**
> * **What's the performance?**
> * **How to install OpenFace?**

---

### **What's the task?**
Design an algorithm to determine whether two face images come from the same person.

### **What's the dataset?**
"webface" is an open dataset, containing 10k+ people’s face images. The folder name can be viewed as person ID. The faces in dataset have been aligned.

### **Which Scheme to choose?**
#### **【Scheme 1】. 利用预训练模型VGG16提取图片bottleneck feature**

 - **VGG16模型：**
    前5组均为卷积层，最后3层为全连接层，该网络在ImageNet数据集上训练所得。本方案中，我们只利用网络的卷积层部分，将全连接以后的部分舍去。
 - **流程：**
    - Step 1. 载入训练好的无全连接层的VGG16网络的权重；
    - Step 2. 输入图片，将得到的输出（bottleneck feature，网络在全连接之前的最后一层激活的feature map）记录到numpy array，再加上每张图片的personID，一起存到离线文件；
    - Step 3. 基于记录下来的特征和personID，我们构成不同的pair，feature是两张图的bottleneck feature，label是0或1（分别表示不是同一人或不是同一个人），正负样本分布基本均匀；
    - Step 4. 基于训练集pair和测试集pair，训练一个全连接分类神经网络。


 - **缺陷**：
    在经过实践后，我们发现利用提取到的bottleneck feature pair训练分类神经网络时，神经网络没办法准确地进行分类，准确率十分低，我们简单计算了不同两张图特征的欧氏距离，发现label为0或1的pair的欧式距离并没有很大的差距，那么毫无疑问是特征的问题了。当我们从源头开始研究VGG16网络时，幡然醒悟，ImageNet数据集！我们利用VGG16得到的图片类别都是人，不能准确地达到同一个人脸得到的特征尽可能相似，而不同人脸得到的特征尽可能不同的目的。

*基于此，我们开始了基于fine-tune的Scheme 2。*

#### **【Scheme 2】. 利用webface图片将ResNet50fine-tune**
- **ResNet50模型：**
    - 为什么改成ResNet？在查找预训练模型的过程中，我们发现了准确率远高于VGG的ResNet，它可以有效地解决“退化“问题，即当模型的层次加深时，错误率却增加了的问题。
    - 	Residual？增加了一个identity mapping（恒等映射），联系到图像处理中的残差向量编码，将原来需要学习的H(x)转换成F(x)+x，使优化的难度下降，起到很好的优化训练的效果。通过shortcut connection实现Residual block，将block的输入和输出进行element-wise的加叠，这个简单的加法并不会给网络增加额外的参数和计算量，同时却可以大大增加模型的训练速度、提高训练效果，并且当模型的层数加深时，这个简单的结构能够很好的解决退化问题。
- **流程：**
    - Step 1. 载入训练好的ResNet50网络的权重作为初始值；
    - Step 2. 只fine-tune最后的卷积块，冻结前170层；
    - Step 3. 将每个文件夹的图片作为输入，personID作为label，训练一个personID个数的分类神经网络。

- **缺陷**：
    经过实践后发现，即使是只训练最后的卷积块，由于为了不破坏原预训练权重，我们的学习率很低，训练速度很慢，尤其是在没有GPU资源的情况下；再加上人脸图片的数据集较大，对于个人PC而言，挑战较大。


#### **【Scheme 3】. 利用openface模型提取图片embedding**
- **说明：**
    这一思路的很大部分来自于https://zhuanlan.zhihu.com/p/24567586
	OpenFace在linux上的安装实录见**How to install OpenFace?**。

- **OpenFace：**
    DCNN。一个深度卷积神经网络，训练其使之输出脸部embedding。但是，并不是让它去识别图片中的物体，而是要让它为脸部生成 128 个测量值。
每次训练要观察三个不同的脸部图像：
1. 加载一张已知的人的面部训练图像
2. 加载同一个人的另一张照片
3. 加载另外一个人的照片

    然后，算法查看它自己为这三个图片生成的测量值。再然后，稍微调整神经网络，以确保第一张和第二张生成的测量值接近，而第二张和第三张生成的测量值略有不同。

- **流程：**
    - Step 1. 利用shape_predictor_68_face_landmarks.dat模型进行面部特征点估计；
    - Step 2. 利用nn4.small2.v1.t7模型，对人脸进行测量，生成128个测量值（Embedding），将每张人脸的embedding以及其personID存储为离线文件；
    - Step 3. 基于记录下来的特征和personID，我们构成不同的pair，feature是两张图的embedding，label是0或1（分别表示不是同一人或不是同一个人），经过对正样本经过**降采样**，使正负样本分布基本均匀；
    - Step 4. 基于训练集pair和测试集pair，训练一个全连接分类神经网络。

#### **综上**
我们选择了准确率更高、个人pc更方便训练的Scheme 3——利用OpenFace提取人脸图片的embedding，再构成pair训练一个全连接的分类神经网络。

### **How to use the code?**

 - openface_train_model.h5：基于训练集pair和测试集pair，训练得到的全连接分类神经网络模型。
 - FaceVerification.py：调用FaceVerification函数的demo文件，路径需要做如下修改：
```python
# 改成openface文件夹下models文件夹路径
modelDir = '/home/xyq/openface/models'
# 下面两行不用改
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# 改成图片1路径
img1Dir = '/home/xyq/PycharmProjects/Openface/006.jpg'
# 改成图片2路径
img2Dir = '/home/xyq/PycharmProjects/Openface/009.jpg'
# 改成分类网络模型路径
modelDir = '/home/xyq/PycharmProjects/Openface/openface_train_model.h5'
```

### **What's the performance?**
训练集选择了4300个人共1600万个piar对,对网络进行训练,训练结果如下：
loss: 0.2532 acc: 0.8970
测试集提取了240万pair对,测试结果如下：
loss: 0.3417 acc: 0.8511

### **How to install OpenFace?**
- 说明：
OpenFace HomePage: https://cmusatyalab.github.io/openface/
OpenFace 安装过程参考：http://blog.csdn.net/weixinhum/article/details/77046873
Linux版本：ubuntu16.04 Kylin

1.	安装git
	sudo apt-get install git

2.	安装编译工具cmake
	sudo apt-get install cmake

3.	安装C++标准库
	sudo agt-get install libboost-dev
	sudo apt-get install libboost-python-dev

4.	安装OpenCV
	sudo apt-get install libopencv-dev
	sudo apt-get install python-opencv

5.	安装Linux安装包管理工具pip
	sudo apt install python-pip
	pip install --upgrade pip

6.	下载OpenFace
	git clone https://github.com/cmusatyalab/openface.git

7.	安装OpenFace依赖的Python库
	cd openface
	sudo pip install -r requirements.txt
	（下载慢可以更换pip源，参考www.cnblogs.com/lqruui/p/6046673.html）
sudo pip install dlib 
	sudo pip install matplotlib

8.	安装luarocks：lua包管理器
sudo apt-get install luarocks

9.	安装Torch
git clone https://github.com/torch/distro.git ~/torch --recursive
	cd torch
	bash install-deps
	./install.sh
	source ~/.bashrc
测试：输入th，如果能进入，则成功，连按两次control+c，退出torch
10.	安装依赖的lua库
luarocks install dpnn
	luarocks install image
	luarocks install nn
	luarocks install graphicsmagick
	luarocks install torchx
	luarocks install csvigo

11.	编译OpenFace
cd ~/openface
	python setup.py build
	sudo python setup.py install

12.	下载预训练的模型
sh models/get-models.sh
wget https://storage.cmusatyalab.org/openface-models/nn4.v1.t7 -O models/openface/nn4.v1.t7

    测试：
cd ~/openface
./demos/compare.py images/examples/{lennon*,clapton*}

openface安装成功！




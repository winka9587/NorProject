# 记录

目录

> -1.
> [link](#test)
> 
> -[to do list](#link_todolist)
##
## network
test

test


## to do list<a id="link_todolist"></a>
### 对照实验
+ 用第t帧作为coord得到的点云，t+1帧作为观测点云pts vs 反过来
  
  理论上用第t帧代替coord的角色会更好，因为初始mask的原因，t帧的点云更干净

### code
1. 主体
> + 训练 <a style="color:red">已完成</a>
>
> + 评估 ：将test数据集输入模型（但是不计算梯度）得到预测结果A1A2（1 OK）后，转换成两帧间的位姿RT（2 OK ）与gt进行比较。得到误差，这里需要误差评估函数（3 working ）
>
> + 给训练添加进度条 working

2. A1A2互逆损失函数 

> <a style="color:red">已完成</a>

3. 可视化工具 

> + open3d可视化代码如何对应到每一个点，例如有深度图和normal map，想将normal map的颜色在点云中可视化出来

4. SGPA中CNN是192x192的

> + 缩小CNN输入的图像尺寸，能否减少CNN的耗时？

6. points_2的点云背景太多

绿色是第一帧的点云,因为有初始的mask,所以能够将被子大致裁剪出来(还是有一定的拖尾),
红色是第二帧的点云,只依赖对初始RGB进行border_add操作得到第二帧的mask，然后裁剪深度图反投影点云。
导致冗余的背景点特别多

<div style="width: 60%;background-color: white">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-1V3QWy.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-A1Mi36.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-Kocxzg.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-UjzahW.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-UblDEJ.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-J86p4J.png' width="30%" >

</div>

> + 用normal map发挥作用，代替分割， 因为第一帧的mask有，但第二帧因为没有mask所以只能靠RGB图像粗略的裁剪来得到点云，但这样会得到大量的背景点云，对于之后计算双向的对应矩阵A是不利的。

> 能否用normalmap来进行粗糙的sift来裁剪点云，并顺应得到的位姿来计算更小的位移delta

6. 对应矩阵A的loss修改

> + 在计算loss的时候有一个问题，SPD中，形变后的deformed prior（称为dpr）与gt模型都处于NOCS坐标系下，因此，将dpr与对应矩阵A相乘后，可以与gt点对点计算误差diff，从而计算loss。
然而对于两帧之间的点云pts1和pts2，就不能这样了，因为二者的点云虽然在相机坐标系下，但是存在着位移t和旋转R的差异，因此倘若直接计算，会导致将t和R引入进而影响loss的值

>一种方法是，每一帧都计算到prior的对应矩阵（但是prior又没有deform过，因此不准确，不过我们先假设prior与instance是完全一致的），

可以直接利用coord map来找到两观测点云的对应关系，从而建立联系。 
下图是序列中连续两帧中同一实例的coord图

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-17-e2Yaov.png' width="100%" >

一种是在coord1和2之间找颜色相同的点，下图中，两帧图像上存在对应关系的点都画为蓝色

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-iHd5l5.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-MehQAS.png' width="30%" >

0:1

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-bguj3B.png' width="100%" >

0:10

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-wux2Qo.png' width="100%" >

0:20

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-MFLiZ9.png' width="100%" >

就算已经间隔了20帧也能够找到对应点,所以可以放心地计算对应关系

7. 既然前面已经找了对应点，只用有对应的点来计算bundle_adjust, 但是对应点并不是完全一样的
该怎么办？

> 能否想办法提高bundle计算A的效率？避免重复计算

8. 提高寻找两张coord图corr的速度 <a style="color:red">已完成</a>
> flatten and reshape 0.0001823902130126953
> 
> to set 3.540173053741455
> 
> intersection 0.00027871131896972656
> 
> append to list 4.513357639312744 (这里可以在to set部分修改来节省时间)
> 
> find correspondence of two coord 8.056186199188232

改用 hash 取代 每个像素点的RGB值 匹配之后速度提高了20多倍

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-19-DWNQrj.png' width="100%" >


<a id="link1"></a>
## 法向

normalspeed来自FFB6D
#### normspeed 参数
```
k_size = 1                      # 单位:pixel  卷积核的大小
distance_threshold = 20000      # 单位:mm     深度值超过这个点云不再参与计算
difference_threshold = 10       # 单位:mm     卷积核中与中心点深度差超过该值则不参与计算
point_into_surface = False
```
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-19-color_mask.gif' width="50%" >

下面是Real数据集中的测试数据
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>

可以看到，其实NOCS数据集的深度差异并不大，FFB6D使用的数据集深度的最大最小值跨度是0~2000左右，
但在NOCS的深度图中，临近的像素深度值往往只有个位数的差异。图中可以看到，尽管 normal map 能够将轮廓大致刻画出来
但如果想更好地将点云分割出来还是需要借助RGB的信息。


下面是Real数据集中的测试数据
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-19-2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg' width="100%" >
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>

###距离问题
像下图中这样，远处的点不参与计算，乍一看以为是 distance_threshold 的问题，
当点的z值超过某一个值就不参与运算了

但实际上是 difference_threshold 的问题，当算子周围的像素Z值超过这个阈值之后就会不参与匀速，
应该是因为远处的点之间的距离会越来越大。

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-23-normal_in_0018__k=1_dist=20000_diff=5_scale_1.0.png' width="100%" >

因此只需要将这个值增大即可，下面是5->50

##基于法向的快速分割网络

能否用一个简单的PSP-Net改一下，可以用合成数据集进行训练，主要是速度必须快。

先去github看看有没有什么现成的网络。直接将RGB信息替换为法向信息能否起作用？
效果是否会更好？速度是否有变化？在周围为白色padding的图片上能否准确分割？

那既然这样，能否直接用网络估计出来对应位置。不，就像估计R改成估计△R一样，
将步骤12拆分成1和2相比12一起做效果会好。

##基于法向的粗点云匹配

利用因为同一物体的法向在相邻两帧之间是大致相同的，
而且我做的并不是点云的匹配，只是为了快速地从下一帧中将所需的点云粗略地提取出来
必须考虑点云丢失的问题，例如上图中的杯子，缺失点云的地方是否是杯子的一部分？这些想用初始mask来弥补
但是在帧与帧之间一定会有损失，必须考虑

我的方法能不能在整个过程中重建模型？（例如在bundle的过程中）重建的模型是否能够反过来为流程提供帮助？
（例如将重建的模型投影来得到粗mask）

考虑上面这些因素，然后考虑一下具体怎么做：

1. 计算normal图，然后用RGB特征匹配对normal图进行分割

2. 用点云分割网络

3. 用RGB网络，但是输入normal图

<p style="color:red">
其实normal map和点云就是法向在不同特征空间下的表现形式，
用哪一种都可以。如果用网络的话，可以单独训练这个网络。
</p>

如果是RGB： 
  输入：nrm1，nrm2，mask1
  测试：mask1裁剪nrm1，然后去nrm2匹配，得到mask2

如果是点云：
  输入：pcd1，pcd2，mask1
  测试：裁剪过的pcd1_mask1，与pcd2的特征融合，得到mask2

或者一种思路：
  将"normal估计下一帧mask"这一过程隐式地包含在对应矩阵估计中，
  即：用normal主导点之间的对应关系

再弄一个损失函数？计算对应点之间的法向？或者对应点之间一定区域内的法向分布？

一个想法：既然CD距离能够计算距离分布，能否在normal空间计算一个CD距离？
CD距离是怎么计算的来着？
CD距离：每一个点，计算另一个点云中与他最近的点；反之也一样。
最终得到一个值，衡量两个点云的相似程度，越小相似度越高。

法向CD距离：将点云中的所有点，将其法向作为xyz，生成法向空间中的点云，称为xyz_n。
对xyz_n1中的每个点，计算在xyz_2中最近点的欧氏距离。反之一样。
最终得到一个值，衡量两组点云相似程度。

问题1：点与点之间并不是一一对应，而是与多个点按一定权重相对应

问题2：这样是否会导致趋向于匹配点云法向为0的点？如果出现了这种情况，法向为NaN的点不参与计算。

先拿两帧的点云测试一下对normal_pcd计算CD的效果如何

<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-0000_color.png' width="100%" >
<p  style="margin-top: 0">real_train/scene_5/0000</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-0010_color.png' width="100%" >
<p style="margin-top: 0">real_train/scene_5/0010</p>
</div>
</div>

有意思的是，点云空间和normal点云空间可以构建对应关系。
下图中的是mask裁剪的点云点的法向点云
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-nrmpcd.gif' width="100%" >
<p  style="margin-top: 0">green:0000, red:0001</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-nrmpcd2.gif' width="100%" >
<p style="margin-top: 0">green:0000, red:0010</p>
</div>
</div>

甚至能找到一条线将其切开

<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-Xe4rVj.png' width="100%" >
<p  style="margin-top: 0">green:0000, red:0001</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-VEhKZJ.png' width="100%" >
<p style="margin-top: 0">green:0000, red:0001</p>
</div>
</div>

但是有问题，下图是0000和0010的法向点云，但是0010的点云是使用0000的mask_add裁剪的，但其计算出的CD loss的值却是0.003994
0000和0001的gt_mask裁剪的法向点云的cd loss都是0.004569，只是因为点更多，导致只要点的数量够多，就有可能出现接近的法向，因为法向一共就那么几个方向。

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-TcJ0JE.png' width="60%" >

那么，使用法向来分割的思路是否可行？
<p style="color:red">
做这个分割的初衷是因为3D-GCN可能不满足跟踪任务对速度的要求，但是本身这个任务很简单，只在一小片区域内寻找目标物体。
</p>
一种思路，normal map训练分割，数据量肯定足够，对于每一个物体，对其深度图进行crop作为原始输入
因为在跟踪中需要的也是输入一个mask过的depth1和一个切割过的depth2得到depth2的mask

另一种思路，是否可以用transformer来做？ (Q法向+K法向) V第一帧mask -> 第二帧mask


法向缺失的部分存疑，可能是物体的一部分，也可能不是。需要额外处理

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-normal_out_0666__k=6_dist=50000_diff=20_scale_1.0.png' width="100%" >

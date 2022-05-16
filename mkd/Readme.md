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
> + 训练 OK
>
> + 评估 ：将test数据集输入模型（但是不计算梯度）得到预测结果A1A2（1 OK）后，转换成两帧间的位姿RT（2 OK ）与gt进行比较。得到误差，这里需要误差评估函数（3 working ）
>
> + 给训练添加进度条 working

2. A1A2互逆损失函数 

> 完成

3. 可视化工具 

> + open3d可视化代码如何对应到每一个点，例如有深度图和normal map，想将normal map的颜色在点云中可视化出来

4. SGPA中CNN是192x192的

> + 缩小CNN输入的图像尺寸，能否减少CNN的耗时？

6. points_2的点云背景太多

绿色是第一帧的点云,因为有初始的mask,所以能够将被子大致裁剪出来(还是有一定的拖尾),
红色是第二帧的点云,只依赖对初始RGB进行border_add操作得到第二帧的mask，然后裁剪深度图反投影点云。
导致冗余的背景点特别多

<div style="width: 60%;background-color: white">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/NorProject_imgs/2022-05-16-1V3QWy.png' width="30%" >
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/NorProject_imgs/2022-05-16-A1Mi36.png' width="30%" >
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/NorProject_imgs/2022-05-16-Kocxzg.png' width="30%" >
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/NorProject_imgs/2022-05-16-UjzahW.png' width="30%" >
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/NorProject_imgs/2022-05-16-UblDEJ.png' width="30%" >
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/NorProject_imgs/2022-05-16-J86p4J.png' width="30%" >

</div>

> + 用normal map发挥作用，代替分割， 因为第一帧的mask有，但第二帧因为没有mask所以只能靠RGB图像粗略的裁剪来得到点云，但这样会得到大量的背景点云，对于之后计算双向的对应矩阵A是不利的。

> 能否用normalmap来进行粗糙的sift来裁剪点云，并顺应得到的位姿来计算更小的位移delta

6. 对应矩阵A的loss修改

> + 在计算loss的时候有一个问题，SPD中，形变后的deformed prior（称为dpr）与gt模型都处于NOCS坐标系下，因此，将dpr与对应矩阵A相乘后，可以与gt点对点计算误差diff，从而计算loss。
然而对于两帧之间的点云pts1和pts2，就不能这样了，因为二者的点云虽然在相机坐标系下，但是存在着位移t和旋转R的差异，因此倘若直接计算，会导致将t和R引入进而影响loss的值

>一种方法是，每一帧都计算到prior的对应矩阵（但是prior又没有deform过，因此不准确，不过我们先假设prior与instance是完全一致的），


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

下面是Real数据集中的测试数据
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>
下面是Real数据集中的测试数据
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>
下面是Real数据集中的测试数据
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>
下面是Real数据集中的测试数据
<div id="test" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://gitee.com/winka9587/markdown-imgs/raw/master/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>




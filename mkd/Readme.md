# ��¼

Ŀ¼

> -1.
> [link](#test)
> 
> -[to do list](#link_todolist)
##
## network
test

test


## to do list<a id="link_todolist"></a>
### ����ʵ��
+ �õ�t֡��Ϊcoord�õ��ĵ��ƣ�t+1֡��Ϊ�۲����pts vs ������
  
  �������õ�t֡����coord�Ľ�ɫ����ã���Ϊ��ʼmask��ԭ��t֡�ĵ��Ƹ��ɾ�

### code
1. ����
> + ѵ�� <a style="color:red">�����</a>
>
> + ���� ����test���ݼ�����ģ�ͣ����ǲ������ݶȣ��õ�Ԥ����A1A2��1 OK����ת������֡���λ��RT��2 OK ����gt���бȽϡ��õ���������Ҫ�������������3 working ��
>
> + ��ѵ����ӽ����� working

2. A1A2������ʧ���� 

> <a style="color:red">�����</a>

3. ���ӻ����� 

> + open3d���ӻ�������ζ�Ӧ��ÿһ���㣬���������ͼ��normal map���뽫normal map����ɫ�ڵ����п��ӻ�����

4. SGPA��CNN��192x192��

> + ��СCNN�����ͼ��ߴ磬�ܷ����CNN�ĺ�ʱ��

6. points_2�ĵ��Ʊ���̫��

��ɫ�ǵ�һ֡�ĵ���,��Ϊ�г�ʼ��mask,�����ܹ������Ӵ��²ü�����(������һ������β),
��ɫ�ǵڶ�֡�ĵ���,ֻ�����Գ�ʼRGB����border_add�����õ��ڶ�֡��mask��Ȼ��ü����ͼ��ͶӰ���ơ�
��������ı������ر��

<div style="width: 60%;background-color: white">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-1V3QWy.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-A1Mi36.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-Kocxzg.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-UjzahW.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-UblDEJ.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-16-J86p4J.png' width="30%" >

</div>

> + ��normal map�������ã�����ָ ��Ϊ��һ֡��mask�У����ڶ�֡��Ϊû��mask����ֻ�ܿ�RGBͼ����ԵĲü����õ����ƣ���������õ������ı������ƣ�����֮�����˫��Ķ�Ӧ����A�ǲ����ġ�

> �ܷ���normalmap�����дֲڵ�sift���ü����ƣ���˳Ӧ�õ���λ���������С��λ��delta

6. ��Ӧ����A��loss�޸�

> + �ڼ���loss��ʱ����һ�����⣬SPD�У��α���deformed prior����Ϊdpr����gtģ�Ͷ�����NOCS����ϵ�£���ˣ���dpr���Ӧ����A��˺󣬿�����gt��Ե�������diff���Ӷ�����loss��
Ȼ��������֮֡��ĵ���pts1��pts2���Ͳ��������ˣ���Ϊ���ߵĵ�����Ȼ���������ϵ�£����Ǵ�����λ��t����תR�Ĳ��죬�������ֱ�Ӽ��㣬�ᵼ�½�t��R�������Ӱ��loss��ֵ

>һ�ַ����ǣ�ÿһ֡�����㵽prior�Ķ�Ӧ���󣨵���prior��û��deform������˲�׼ȷ�����������ȼ���prior��instance����ȫһ�µģ���

����ֱ������coord map���ҵ����۲���ƵĶ�Ӧ��ϵ���Ӷ�������ϵ�� 
��ͼ��������������֡��ͬһʵ����coordͼ

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-17-e2Yaov.png' width="100%" >

һ������coord1��2֮������ɫ��ͬ�ĵ㣬��ͼ�У���֡ͼ���ϴ��ڶ�Ӧ��ϵ�ĵ㶼��Ϊ��ɫ

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-iHd5l5.png' width="30%" >
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-MehQAS.png' width="30%" >

0:1

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-bguj3B.png' width="100%" >

0:10

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-wux2Qo.png' width="100%" >

0:20

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-18-MFLiZ9.png' width="100%" >

�����Ѿ������20֡Ҳ�ܹ��ҵ���Ӧ��,���Կ��Է��ĵؼ����Ӧ��ϵ

7. ��Ȼǰ���Ѿ����˶�Ӧ�㣬ֻ���ж�Ӧ�ĵ�������bundle_adjust, ���Ƕ�Ӧ�㲢������ȫһ����
����ô�죿

> �ܷ���취���bundle����A��Ч�ʣ������ظ�����

8. ���Ѱ������coordͼcorr���ٶ� <a style="color:red">�����</a>
> flatten and reshape 0.0001823902130126953
> 
> to set 3.540173053741455
> 
> intersection 0.00027871131896972656
> 
> append to list 4.513357639312744 (���������to set�����޸�����ʡʱ��)
> 
> find correspondence of two coord 8.056186199188232

���� hash ȡ�� ÿ�����ص��RGBֵ ƥ��֮���ٶ������20�౶

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-19-DWNQrj.png' width="100%" >


<a id="link1"></a>
## ����

normalspeed����FFB6D
#### normspeed ����
```
k_size = 1                      # ��λ:pixel  ����˵Ĵ�С
distance_threshold = 20000      # ��λ:mm     ���ֵ����������Ʋ��ٲ������
difference_threshold = 10       # ��λ:mm     ������������ĵ���Ȳ����ֵ�򲻲������
point_into_surface = False
```
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-19-color_mask.gif' width="50%" >

������Real���ݼ��еĲ�������
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

���Կ�������ʵNOCS���ݼ�����Ȳ��첢����FFB6Dʹ�õ����ݼ���ȵ������Сֵ�����0~2000���ң�
����NOCS�����ͼ�У��ٽ����������ֵ����ֻ�и�λ���Ĳ��졣ͼ�п��Կ��������� normal map �ܹ����������¿̻�����
���������õؽ����Ʒָ����������Ҫ����RGB����Ϣ��


������Real���ݼ��еĲ�������
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

###��������
����ͼ��������Զ���ĵ㲻������㣬էһ����Ϊ�� distance_threshold �����⣬
�����zֵ����ĳһ��ֵ�Ͳ�����������

��ʵ������ difference_threshold �����⣬��������Χ������Zֵ���������ֵ֮��ͻ᲻�������٣�
Ӧ������ΪԶ���ĵ�֮��ľ����Խ��Խ��

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-05-23-normal_in_0018__k=1_dist=20000_diff=5_scale_1.0.png' width="100%" >

���ֻ��Ҫ�����ֵ���󼴿ɣ�������5->50

##���ڷ���Ŀ��ٷָ�����

�ܷ���һ���򵥵�PSP-Net��һ�£������úϳ����ݼ�����ѵ������Ҫ���ٶȱ���졣

��ȥgithub������û��ʲô�ֳɵ����硣ֱ�ӽ�RGB��Ϣ�滻Ϊ������Ϣ�ܷ������ã�
Ч���Ƿ����ã��ٶ��Ƿ��б仯������ΧΪ��ɫpadding��ͼƬ���ܷ�׼ȷ�ָ

�Ǽ�Ȼ�������ܷ�ֱ����������Ƴ�����Ӧλ�á������������R�ĳɹ��ơ�Rһ����
������12��ֳ�1��2���12һ����Ч����á�

##���ڷ���Ĵֵ���ƥ��

������Ϊͬһ����ķ�����������֮֡���Ǵ�����ͬ�ģ�
���������Ĳ����ǵ��Ƶ�ƥ�䣬ֻ��Ϊ�˿��ٵش���һ֡�н�����ĵ��ƴ��Ե���ȡ����
���뿼�ǵ��ƶ�ʧ�����⣬������ͼ�еı��ӣ�ȱʧ���Ƶĵط��Ƿ��Ǳ��ӵ�һ���֣���Щ���ó�ʼmask���ֲ�
������֡��֮֡��һ��������ʧ�����뿼��

�ҵķ����ܲ����������������ؽ�ģ�ͣ���������bundle�Ĺ����У��ؽ���ģ���Ƿ��ܹ�������Ϊ�����ṩ������
�����罫�ؽ���ģ��ͶӰ���õ���mask��

### padding����

��Ȼ��һƪ������֤��padding����Ӱ��CNN������������ǲ���Ӱ���ٶȡ�ͬʱ��
paddingҲ��Ӱ�����normal map���ٶȡ�

<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-09-norm.PNG' width="100%" >
<p  style="margin-top: 0">�ü�������ͼ</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-09-norm2.PNG' width="100%" >
<p style="margin-top: 0">���ɵ�normal map</p>
</div>
</div>

compute normal map (cropped) 0.004948854446411133

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-09-norm_padding.PNG' width="100%" >

compute normal map (padding) 0.023913860321044922

���Է���padding�ļ���ʱ������Ҫ��cropped��

### �ü����ͼ�Ƿ��Ӱ��normalspeed���ɵķ���ͼ��

ԭ���ͼ

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-18-CKKcNJ.png' width="70%" >

�������ͼ���ɵķ���ͼ nrm_full

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-18-EhGsgF.png' width="70%" >

�ü������ͼ���ɵķ���ͼ nrm_crop

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-18-YdmVLn.png' width="40%" >

����ǰ��ķ���ͼ��һ������Ϊû�����ӻ�������nrm_full���вü����ü����nrm_full��nrm_crop����õ���ͼ��
���Է��֣�ֻ��ͼ���Ե�в��죬������������ͬ�ġ�

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-18-mnvRvN.png' width="40%" >


### Now

����������Щ���أ�Ȼ����һ�¾�����ô����

1. ����normalͼ��Ȼ����RGB����ƥ���normalͼ���зָ�

2. �õ��Ʒָ�����

3. ��RGB���磬��������normalͼ

<p style="color:red">
��ʵnormal map�͵��ƾ��Ƿ����ڲ�ͬ�����ռ��µı�����ʽ��
����һ�ֶ����ԡ����������Ļ������Ե���ѵ��������硣
</p>

�����RGB�� 
  ���룺nrm1��nrm2��mask1
  ���ԣ�mask1�ü�nrm1��Ȼ��ȥnrm2ƥ�䣬�õ�mask2

����ǵ��ƣ�
  ���룺pcd1��pcd2��mask1
  ���ԣ��ü�����pcd1_mask1����pcd2�������ںϣ��õ�mask2

����һ��˼·��
  ��"normal������һ֡mask"��һ������ʽ�ذ����ڶ�Ӧ��������У�
  ������normal������֮��Ķ�Ӧ��ϵ

��Ūһ����ʧ�����������Ӧ��֮��ķ��򣿻��߶�Ӧ��֮��һ�������ڵķ���ֲ���

һ���뷨����ȻCD�����ܹ��������ֲ����ܷ���normal�ռ����һ��CD���룿
CD��������ô��������ţ�
CD���룺ÿһ���㣬������һ����������������ĵ㣻��֮Ҳһ����
���յõ�һ��ֵ�������������Ƶ����Ƴ̶ȣ�ԽС���ƶ�Խ�ߡ�

����CD���룺�������е����е㣬���䷨����Ϊxyz�����ɷ���ռ��еĵ��ƣ���Ϊxyz_n��
��xyz_n1�е�ÿ���㣬������xyz_2��������ŷ�Ͼ��롣��֮һ����
���յõ�һ��ֵ����������������Ƴ̶ȡ�

����1�������֮�䲢����һһ��Ӧ�����������㰴һ��Ȩ�����Ӧ

����2�������Ƿ�ᵼ��������ƥ����Ʒ���Ϊ0�ĵ㣿����������������������ΪNaN�ĵ㲻������㡣

������֡�ĵ��Ʋ���һ�¶�normal_pcd����CD��Ч�����

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

����˼���ǣ����ƿռ��normal���ƿռ���Թ�����Ӧ��ϵ��
��ͼ�е���mask�ü��ĵ��Ƶ�ķ������
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

�������ҵ�һ���߽����п�

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

���������⣬��ͼ��0000��0010�ķ�����ƣ�����0010�ĵ�����ʹ��0000��mask_add�ü��ģ�����������CD loss��ֵȴ��0.003994
0000��0001��gt_mask�ü��ķ�����Ƶ�cd loss����0.004569��ֻ����Ϊ����࣬����ֻҪ����������࣬���п��ܳ��ֽӽ��ķ�����Ϊ����һ������ô��������

<p style="color:cornflowerblue">
һ���뷨���ܲ��ܽ�normal_pcd��pcd�������һ���ǣ����յõ��ĵ��Ƽ�Ҫ��ŷʽ�ռ������ǰһ֡������״���ƣ�
��Ҫ�ڷ���ռ�����normal_pcd�ķֲ����ơ�
</p>

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-TcJ0JE.png' width="60%" >

��ô��ʹ�÷������ָ��˼·�Ƿ���У�
<p style="color:red">
������ָ�ĳ�������Ϊ3D-GCN���ܲ��������������ٶȵ�Ҫ�󣬵��Ǳ����������ܼ򵥣�ֻ��һСƬ������Ѱ��Ŀ�����塣
</p>
һ��˼·��normal mapѵ���ָ�������϶��㹻������ÿһ�����壬�������ͼ����crop��Ϊԭʼ����
��Ϊ�ڸ�������Ҫ��Ҳ������һ��mask����depth1��һ���и����depth2�õ�depth2��mask

��һ��˼·���Ƿ������transformer������ (Q����+K����) V��һ֡mask -> �ڶ�֡mask

����ȱʧ�Ĳ��ִ��ɣ������������һ���֣�Ҳ���ܲ��ǡ���Ҫ���⴦��

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-02-normal_out_0666__k=6_dist=50000_diff=20_scale_1.0.png' width="100%" >

to do��

������һ�����ݼ�
��pointnet2���ָ̫���ˣ����ã������������3D-GCN���ο�SAR-Net��

SAR-Net��ѵ����ʱ�������ֻ��һ��mask����ȡ� ��������֧��Ϊ�˴���ͬ��Ŀ���ⷽ����
��Ϊ�е�Ŀ���ⷽ���ǵõ����ص㣬�е��ǵõ�bbox��
��SAR-Net�е�3D-GCN��ͬ���ǣ���Ϊ���Ǹ�����������������ǰһ֡��mask�ġ�
<p style="color: deepskyblue">
˼·��ʹ��Ŀ�����㷨��Ȼ����3D-GCN�õ����ơ���No��
�������򱾾Ͳ��󣬿���ֱ�ӵ���bbox�ļ���������޸�3D-GCN������ǰһ֡����Ϣ���õ���ǰ֡�ĵ���
</p>
˳������������normal map��2D��ɫ�ֲ������ٶ�λ��ǰ֡���������ͼ���п��ܵ�λ�ã���������2D����
��sift��ʱ̫���ˣ�

> extract kp end 0.019520282745361328
>
> matcher end 0.0
>
> match end 0.0009765625

3D-GCN���ٶ����⣬�ȿ���ȡ�ٽ�����������Ӧ���ǳ������ȽϺ�ʱ�Ĳ�������3dgcn���д��3��

> ��4096��������ȡ50���ٽ���
>
> get Neighbor 0.06122946739196777
>
> get Neighbor 0.05437242984771729
> 
> 1024������ȡ10���ٽ���,�ٶȻ����Խ��ܣ�����ֵ������鲻ͬ����ʱ����ܴ�
> 
> get Neighbor 0.004289388656616211
> 
> get Neighbor 0.0013625621795654297

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-06-CusnEP.png' width="100%" >


SAR-Net��������й���3D-GCN�����ã�

�ָ������3D-GCN�ٶȲ��� ���鲻ͬ��1024����ĵ��Ʒֱ���ԡ��ٽ�����20�������շָ������Ϊ2

> get Neighbor 0.23159170150756836
> 
> get Neighbor 0.2432103157043457

�ٽ�����10�������շָ������Ϊ2

> get Neighbor 0.0960237979888916
> 
> get Neighbor 0.1004629135131836

�ٽ�����5�������շָ������Ϊ2

> get Neighbor 0.030435562133789062
> 
> get Neighbor 0.05418968200683594
> 
����ֻ��5���ٽ��㣬��ʱҲ��̫�ܽ��ܡ���Ҫ�������������

<p style="color:deepskyblue;">
ֵ��һ�ᣬ3D-GCN���ᵽ������ȡ����ʱ����ѡ�����꣬����RGB�ȡ��ڷָ�ʱ�ܷ���xyz+normal��
</p>


### Ŀǰ��ʧ������

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-13-c6J1Tr.png' width="70%" >

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-13-xSiOJY.png' width="70%" >

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-13-aAGYof.png' width="70%" >

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-13-OwtaNS.png' width="70%" >


����ԭ�������mask�ü����ɾ�������β��ѧϰ������Ӱ�죬���⣬��û��ʲô�취�ܹ����ӻ���Ӧ��ϵ��

���ӻ�һ��ѵ��ʱ�Ķ�Ӧ���������

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-13-Ba5KRr.png' width="100%" >

�ư�������ѵ�����Ķ�Ӧ��֤����one_hot����ʽ����˵õ�pts1��������ܵ�����pts2���Ӱ�죬��˻���������ƽ��ֵһ��
���������������loss,����1000(entropy loss to encourage peaked distribution)

���⣺����loss����loss������Ϊ0����ζ�����еĶ�Ӧ�����ÿһ�ж���Լ������one-hot������
������Ϊloss�в�û��Ҫ�����е�one-hot���⣬���Ե������е�one-hot��������������ͬһ�㡣
�����е�pts1�еĵ㶼ӳ�䵽��pts2�ϵ�ͬһ���㡣

Ӧ������loss̫���µģ�1.���Խ����С�� 2.�ܷ�����lossԼ��one-hot����֮�以�⣿

�ƺ�������loss�����⣬����Ȩ����1����loss����ʱ���õ�����Ȼ�����е�ӳ�䵽һ����

1.���ڳ��Խ�����lossҲ���ͣ����߽�����lossɾ����һ�ԡ���No�����ӻ���loss��Ȼ��ӳ�䵽һ���㣬
���Ϸ�������һ���ߣ�
2.Ȼ�����Լ�����е�one-hot���⣨���ڵ���lossֻ��ˮƽԼ�������� ����˵ӳ�䵽ͬһ����ĵ����������ƣ���ӳ��������ɾ������
3.��һ����Ӧ����ĳ�ֵ��

<p style="color:deepskyblue;">
һ�����ʣ�Ϊʲô������һ�㷴��loss��С������������corr_loss��Ŀ�ı��������ˣ�

����gt��corr_loss��������ʲô����
</p>

�����ģ���ܹ���һ������ĳ�ֵ�������Ӧ����������ĵ㡣

���ǽ�CNN�������padding�滻Ϊbbox�ü�֮��Ľ������֮ǰ������һЩ��������Ȼû�ܽ������

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-20-aWyo3y.png' width="100%" >


### ������ʧ����corr_loss
sRt12_gt��gtλ�˱任���ܹ���points_1�ĵ�ͨ������任�䵽points2��λ�á���˱任��ĵ���points_1�ĵ㻹���ж�Ӧ��ϵ�ġ�
��˿������Ӧ����soft_assign_1ӳ���ĵ�������������

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-14-qiMmR5.png' width="100%" >

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-14-KwU7Q9.png' width="70%" >

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-14-XQ6Cez.png' width="70%" >

Ϊʲô������һС���ֵ�ӳ�����ܱ�ԭ���Ƶ�����С��

��CD loss��Լ����״��CD loss��λ���йأ������Թ�һ�����ټ���loss��

������Χ����ֱ��ȥԤ���Ӧ���Ƿ���У��� 

���һ�²�ͬ������ֵ�򣬲�ͬ����concat֮ǰ�Ƿ���Ҫ��һ����

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-15-G4Sin9.png' width="100%" >

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-15-f3J2Nu.png' width="100%" >






<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-15-ImzKxp.png' width="100%" >
<p  style="margin-top: 0">SGPA��eval��ʵ������</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-15-q2HLaS.png' width="100%" >
<p style="margin-top: 0">��Ӧ����*ʵ������</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-15-vYpLIr.png' width="100%" >
<p style="margin-top: 0">SGPA��eval�й۲����</p>
</div>
</div>

��ͼ�У���Ӧ���������õĲ����ǵ�1,2ͼ����ΪҪ�ǵ�ʵ�����ƣ���prior+D���εõ��ģ����γɽӽ�gtģ�ͣ���ȱ��������

## SAR-Net��ѵ��3D-GCN

### ����

ʹ�úϳ����ݼ�CAMERA������ѵ�����ݣ���Ϊ��CAMERAֱ�ӽ�����������Ⱦ����ʵ�����ϣ�
���Բ�������ʵ���ݼ���mask_depth��ͶӰ����β������

SAR-Net�����������bbox��mask���ֲ�ͬ�ķָ�������ѵ����

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-20-tkp8nv.png' width="100%" >

��gt���д���ʹ������������أ�

(1)bbox�ָ������ֱ��ƶ����ϽǺ����Ͻ�����bbox���б任���ƶ���Χ��-5~15����

(2)mask�ָ�������mask����0~5�����أ����Կ׶�(������˱��İ��ֽ������)

### �뷨����һ�źϳ����ݼ���normal map�����Ƿ��ǰ�����ָ��ܹ���������


### ����lossʱ�ƺ�������

<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/Norproject/2022-06-23-LSxZ3a.png' width="50%" >

��ͼ��4�����,

��ɫ:     pts2to1   # pts2ͨ��gtλ�˱任��1
��ɫ:     pts1to2   # pts1ͨ��gtλ�˱任��2
��ɫ������ɫ:   pts1��pts2

gtλ��
(-0.0004, -0.0023, -0.0004)
����
(-0.0528,  0.0318,  0.0375)

Ӧ���� multi_pose_with_pts �������⣿����pts1��pts2��ƽ����������Ҳ�ر�󡣿��ӻ�һ��ƽ����

�ƺ��ҵ�ԭ����
λ��tû�����⣬������ֻҪ������ת����R���ͻᷢ��һ����λ�ơ�

�Ƿ��Ǿ��ȵ��µ����⣿get_total_loss_2_frame�Ĳ���pose12_gt��float32��

t��R�Ĺ�ʽ�ƴ��ˣ������ֽ

��������Ȼ���У�����֡ͶӰ��ͼ���ϡ���֤�ˣ���nocs��camera�ı任��û������ģ�����sRt1��sRt2�任�ĵ������غϵġ�
���Բ���������ͼ���������Զ

��model,sRT*model��۲����pts��������ͬʱ���Ƴ������ƺ����и������

<a style="color:blue">
[ȷ����һ���£�gt_part�ļ����ǲ���Ҫ�ģ�ֻ��nocs2camera������ ��������ֵ��ȫ��һ����]
</a>

��CAPTRA�Ĳ��Դ����У�objģ��ͨ��sRT������pts�غϣ�ֻ�ǲ��scale

����NorPorjec�У�objģ��ͨ��sRT֮����pts����ر�Զ��Ϊʲô��������ֱ�������������ͬһ��ͼƬ�������ǵõ��Ľ��ۣ�
  1.λ��sRt��һ���ģ����ٴ�����һ���ģ�������һЩ���������ڴ��򶼲�����
  2.Ŀǰ�뵽�Ŀ��ܵ�ԭ���ǹ۲����pts�����⡣��һ��CAPTRA������pts����εõ��ġ�

>  pts, idxs = backproject(depth, intrinsics, mask == inst_i)

��backproject�жԵõ����Ƶ�z�ܽ�����ȡ������������ȡ��֮��Ҳ���С�
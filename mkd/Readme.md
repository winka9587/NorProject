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

> <p style="color:red">�����</p>

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

������Real���ݼ��еĲ�������
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>

���Կ�������ʵNOCS���ݼ�����Ȳ��첢����FFB6Dʹ�õ����ݼ���ȵ������Сֵ�����0~2000���ң�
����NOCS�����ͼ�У��ٽ����������ֵ����ֻ�и�λ���Ĳ��졣ͼ�п��Կ��������� normal map �ܹ����������¿̻�����
���������õؽ����Ʒָ����������Ҫ����RGB����Ϣ��


������Real���ݼ��еĲ�������
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>
������Real���ݼ��еĲ�������
<div class="img_group" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>
������Real���ݼ��еĲ�������
<div id="test" style="text-align:center;">
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-depth_view_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">depth_view</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_in_0000__k=1_dist=20000_diff=10.jpg'/>
<p style="margin-top: 0">normal_map_in</p>
</div>
<div class="sub_img" style="width:30%;display: inline-block;">
<img src='https://raw.githubusercontent.com/winka9587/MD_imgs/main/2022-05-10-normal_out_0000__k=1_dist=20000_diff=10.jpg'/>
<p  style="margin-top: 0">normal_map_out</p>
</div>
</div>




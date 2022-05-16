# comet.ml安装
## 1.install

    pip install comet_ml

github页面上说支持python2.7-3.6

## 2.在pytorch代码中添加
[官方pytorch示例](https://github.com/comet-ml/comet-examples/tree/master/pytorch)

初始化 Experiment对象

    experiment = Experiment(project_name="pytorch")

使用experiment.log_metric()和experiment.log_parameters()记录超参数,评估以及可视化

例如： 打印超参

    experiment.log_parameters(hyper_params)


# comet.ml��װ
## 1.install

    pip install comet_ml

githubҳ����˵֧��python2.7-3.6

## 2.��pytorch���������
[�ٷ�pytorchʾ��](https://github.com/comet-ml/comet-examples/tree/master/pytorch)

��ʼ�� Experiment����

    experiment = Experiment(project_name="pytorch")

ʹ��experiment.log_metric()��experiment.log_parameters()��¼������,�����Լ����ӻ�

���磺 ��ӡ����

    experiment.log_parameters(hyper_params)


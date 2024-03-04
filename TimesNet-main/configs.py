import argparse
parser = argparse.ArgumentParser(description='TimesNet')
#模型配置

parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]') #任务名称
parser.add_argument('--model', type=str, default='TimesNet',
                        help='model') #任务名称
# forecasting task
parser.add_argument('--seq_len', type=int, default=3, help='input sequence length') #序列长度 用前seq_len的数据预测
parser.add_argument('--label_len', type=int, default=1, help='start token length') #目标长度 希望模型预测的未来时间步长的数量
parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length') #预测长度 希望模型预测未来 pred_len 天的数据
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#freq 用于时间特征编码，选项：[s:秒，t:分钟，h:小时，d:日，b:工作日，w:周，m:月]，您也可以使用更详细的 freq，如 15 分钟或 3 小时

#data loader
parser.add_argument('--root_path', type=str, default=r'T:\master\TimesNet', help='root_path')  #数据根目录
parser.add_argument('--data_path', type=str, default='煤炭总表.csv', help='data_path')  #数据路径
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#预测任务，选项：[M, S, MS]；M:多变量预测多变量，S:单变量预测单变量，MS:多变量预测单变量

#参数设置
parser.add_argument('--train_epochs', type=int, default=500)  #训练轮数
parser.add_argument('--learning_rate', type=float, default=1e-4)  #学习率
parser.add_argument('--scale', type=bool, default=True,help='Whether to perform feature scaling')  #是否进行特征缩放
parser.add_argument('--batch_size', type=int, default=4)  #批量
parser.add_argument('--num_workers', type=int, default=0)  #
parser.add_argument('--patience', type=int, default=20)  #提前终止轮数
parser.add_argument('--itr', type=int, default=1)  #实验次数
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# model define
parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock')  #取傅里叶相值的前 top_k数据
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception') #Inception模型的参数
parser.add_argument('--enc_in', type=int, default=18, help='encoder input size') #编码器输入尺寸 模型输入的特征数(维度)
parser.add_argument('--dec_in', type=int, default=18, help='decoder input size') #解码器输入尺寸
parser.add_argument('--c_out', type=int, default=18, help='output size') #输出尺寸
parser.add_argument('--d_model', type=int, default=32, help='dimension of model') #模型维度
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers') #编码层个数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') #解码层个数
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn') # FCN 网络中的隐藏层神经元的数量
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') #时间特征编码, 选项:[timeF, 固定, 学习]
args = parser.parse_args()

print('Args in experiment:')
print(args)
#

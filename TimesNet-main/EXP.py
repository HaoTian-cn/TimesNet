from TimesNet import *
from configs import *
from tools import EarlyStopping
from Warehouse import *
from timefeatures import time_features
#数据集
class Dataset_loader(Dataset):
    def __init__(self,args, flag='train',
                 target='OT',  timeenc=1, seasonal_patterns=None):
        '''
        root_path: 数据文件的根路径。
        flag: 数据集的标志，可以是 'train'、'test' 或 'val'，表示训练集、测试集或验证集。
        size: 数据序列的长度信息，格式为 [seq_len, label_len, pred_len]。
        features: 特征的类型，可能是 'S'、'M' 或 'MS'。
        data_path: 数据文件的路径。
        target: 目标特征的名称。
        scale: 是否进行特征缩放。
        timeenc: 时间编码类型，可能是 0 或 1。
        freq: 数据的时间频率。
        seasonal_patterns: 季节性模式，可能是一个序列


        '''
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = args.features
        self.target = target
        self.scale = args.scale
        self.timeenc = timeenc
        self.freq = args.freq #时间编码
        self.scaler = StandardScaler() #归一化
        self.df_raw = pd.read_csv(os.path.join(args.root_path,
                                          args.data_path),encoding='gbk') #载入数据
        border1s =  [0, 10, 15] #上边界 ,数据比较少，全部拿来测试

        border2s = [15,15,20] #下边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            cols_data = self.df_raw.columns[1:]
            df_data = self.df_raw[cols_data]
        else:
            df_data = self.df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = self.df_raw[['date']][border1:border2] #时间列
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        '''
        如果 self.timeenc 的值为 0，意味着选择时间特征的编码方式为按照月、日、星期几和小时。
        如果 self.timeenc 的值为 1，意味着选择另一种时间特征编码方式。在这种情况下，调用了一个名为 time_features 的函数，
        并传递了时间戳数据（转换为 pd.to_datetime() 的结果）以及 freq 参数。函数 time_features 的具体实现不在你提供的代码片段中，
        但是它可能根据给定的时间戳数据生成不同的时间特征表示。然后，通过 .transpose(1, 0) 方法对生成的特征进行转置，使得特征向量在行上，时间点在列上。
        '''
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)

            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

#实验
def data_provider(args, flag):

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid



    drop_last = False
    data_set = Dataset_loader(
        args=args,
        flag=flag
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader

class Exp_Long_Term_Forecast():
    def __init__(self, args):
        self.args = args
        self.model= Model(configs=self.args).float()
        self.model=self.model.to(device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion=nn.MSELoss()
        self.criterion=self.criterion.to(device)
        self.device=device
    def get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    #验证过程
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    def train(self,setting):
        train_data, train_loader = self.get_data(flag='train')
        vali_data, vali_loader = self.get_data(flag='val')
        test_data, test_loader = self.get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        time_now = time.time()
        train_steps = len(train_loader)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                loss.backward()
                self.model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion)
            test_loss = self.vali(test_data, test_loader, self.criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model.state_dict(),best_model_path)
        return self.model


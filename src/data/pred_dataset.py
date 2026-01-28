import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.timefeatures import time_features
import warnings
import random
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False,half=0,all=0,one_channel=0
                 ):
        
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.half =half
        self.all = all
        self.one_channel=one_channel
        self.split = split
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0 * 30 * 24 ,  12 * 30 * 24 - self.seq_len,  12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,  12 * 30 * 24 + 4 * 30 * 24,  12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
        if self.all :
            border1 = 0
            border2 = len(df_raw)
        print(border1, border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

    def __getitem__(self, index):
        s_begin = index
        if self.split == 'train':
            s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # if self.scale:
        #     self.scaler.fit(seq_x)
        #     seq_x = self.scaler.transform(seq_x)
        #     seq_y = self.scaler.transform(seq_y)
        #     seq_x, seq_y = filter_data(seq_x, seq_y)

        if self.split == 'train':
            seq_x, seq_y = channel_mixing(seq_x, seq_y)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):

        if self.split == 'train':
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False,half=0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.split =  split
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [int(0 * 30 * 24 * 4), int(12 * 30 * 24 * 4 - self.seq_len),int( 12 * 30 * 24 * 4 + 4 * 30 * 24 *4 - self.seq_len)]
        border2s = [int(12 * 30 * 24 * 4), int(12 * 30 * 24 * 4+ 4 * 30 * 24 *4), int(12 * 30 * 24 * 4 + 8 * 30 * 24 * 4)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))

        if self.all :
            border1 = 0
            border2 = len(df_raw)
            print(border1,border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        if self.split == 'train':
            s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # if self.split == 'train':
        #     seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.split == 'train':
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y


class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.8, test_split=0.1, half =0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.split =  split

        self.all = all
        # train test ratio
        if self.all:
            self.train_split, self.test_split = 1, 0
        else:
            self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
          
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # data = data[:,-1].reshape(-1,1)
        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

    def __getitem__(self, index):
        s_begin = index

        # if self.split == 'train':
        #     s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.split == 'train':
            seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.split == 'train':
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        # else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y
        

class Dataset_TrainTestCSV(Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        size=None,
        features='S',
        train_path=None,
        test_path=None,
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        time_col_name='Cycle',
        unit_col_name='UnitNumber',
        use_time_features=False,
        train_split=0.9,
        half=0,
        all=0,
        one_channel=0,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert split in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.unit_col_name = unit_col_name
        self.use_time_features = use_time_features
        self.split = split

        self.root_path = root_path
        self.train_path = train_path
        self.test_path = test_path
        self.train_split = train_split
        self.half = half
        self.all = all
        self.one_channel = one_channel

        if self.train_path is None or self.test_path is None:
            raise ValueError('train_path and test_path must be provided')

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        def read_csv_flexible(path):
            df = pd.read_csv(path)
            if self.unit_col_name in df.columns and self.time_col_name in df.columns:
                return df
            df = pd.read_csv(path, header=None)
            n_cols = df.shape[1]
            if n_cols < 3:
                raise ValueError(f'CSV has too few columns: {path}')
            columns = [self.unit_col_name, self.time_col_name]
            for i in range(2, n_cols - 1):
                columns.append(f'feat{i-1}')
            columns.append('RUL')
            df.columns = columns
            return df

        train_file = os.path.join(self.root_path, self.train_path)
        test_file = os.path.join(self.root_path, self.test_path)
        df_train = read_csv_flexible(train_file)
        df_test = read_csv_flexible(test_file)

        df_train = df_train.sort_values([self.unit_col_name, self.time_col_name]).reset_index(drop=True)
        df_test = df_test.sort_values([self.unit_col_name, self.time_col_name]).reset_index(drop=True)

        def to_xy_df(df):
            if self.unit_col_name not in df.columns or self.time_col_name not in df.columns:
                raise ValueError('unit/time columns missing')
            df_no_unit = df.drop(columns=[self.unit_col_name])
            if self.features in ['M', 'MS']:
                x_df = df_no_unit[df_no_unit.columns[1:]]
            else:
                x_df = df_no_unit[[self.target]]
            t = df[self.time_col_name].to_numpy()
            return x_df, t

        x_train_df, t_train = to_xy_df(df_train)
        x_test_df, t_test = to_xy_df(df_test)

        if self.scale:
            self.scaler.fit(x_train_df.values)
            x_train = self.scaler.transform(x_train_df.values)
            x_test = self.scaler.transform(x_test_df.values)
        else:
            x_train = x_train_df.values
            x_test = x_test_df.values

        def build_windows(use_all):
            self._unit_data = []
            self._unit_stamp = [] if self.use_time_features else None
            self._windows = []

            def add_unit_series(unit_x, unit_t, is_train_file, unit_id):
                if self.split == 'test':
                    if is_train_file:
                        return
                    seg_x = unit_x
                    seg_t = unit_t
                else:
                    if not is_train_file:
                        return
                    if use_all:
                        seg_x = unit_x
                        seg_t = unit_t
                    else:
                        cut = int(len(unit_x) * self.train_split)
                        if self.split == 'train':
                            seg_x = unit_x[:cut]
                            seg_t = unit_t[:cut]
                        else:
                            start = max(cut - self.seq_len, 0)
                            seg_x = unit_x[start:]
                            seg_t = unit_t[start:]

                n = len(seg_x) - self.seq_len - self.pred_len + 1
                if n <= 0:
                    return

                start_k = 0
                if self.half != 0 and self.split == 'train':
                    start_k = int(n * (1 - self.half))
                    start_k = min(max(start_k, 0), n)

                unit_index = len(self._unit_data)
                self._unit_data.append(seg_x)
                if self.use_time_features:
                    self._unit_stamp.append(seg_t.reshape(-1, 1))

                for k in range(start_k, n):
                    self._windows.append((unit_index, k))

            for unit_id, df_u in df_train.groupby(self.unit_col_name, sort=True):
                unit_x = x_train[df_u.index.values]
                unit_t = t_train[df_u.index.values]
                add_unit_series(unit_x, unit_t, is_train_file=True, unit_id=unit_id)

            for unit_id, df_u in df_test.groupby(self.unit_col_name, sort=True):
                unit_x = x_test[df_u.index.values]
                unit_t = t_test[df_u.index.values]
                add_unit_series(unit_x, unit_t, is_train_file=False, unit_id=unit_id)

        build_windows(self.all)
        if (len(self._unit_data) == 0 or len(self._windows) == 0) and self.split != 'test' and not self.all:
            self.all = 1
            build_windows(True)

        if len(self._unit_data) == 0 or len(self._windows) == 0:
            raise ValueError('No available windows for the given seq/pred lengths')

        self.number_of_channels = self._unit_data[0].shape[1]

    def __getitem__(self, index):
        unit_index, s_begin = self._windows[index]
        data = self._unit_data[unit_index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data[s_begin:s_end]
        seq_y = data[r_begin:r_end]

        if self.split == 'train':
            seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x, seq_y = self.channel_chosing(seq_x, seq_y, 1)

        if self.use_time_features:
            stamp = self._unit_stamp[unit_index]
            seq_x_mark = stamp[s_begin:s_end]
            seq_y_mark = stamp[r_begin:r_end]
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self._windows)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def channel_chosing(self, seq_x, seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0, C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:, selected_numbers], seq_y[:, selected_numbers]
        return seq_x, seq_y



def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)

def filter_data(seq_x, seq_y):
    """
    Filter function to set seq_x and seq_y to zero arrays 
    if any absolute value in seq_x or seq_y is greater than 9.
    """
    if np.any(np.abs(seq_x) > 9) or np.any(np.abs(seq_y) > 9):
        return np.zeros_like(seq_x), np.zeros_like(seq_y)
    return seq_x, seq_y

def channel_mixing(seq_x, seq_y):
    T, C = seq_x.shape
    L, _ = seq_y.shape
    
    # 1. 通道洗牌
    shuffle_indices = np.random.permutation(C)
    
    # 2. 生成随机值
    random_values = np.random.normal(0, 1, C)
    
    # 3. 通道调整
    shuffled_seq_x = seq_x[:, shuffle_indices]
    shuffled_seq_y = seq_y[:, shuffle_indices]
    
    # 4. 乘以随机值
    mixed_seq_x = shuffled_seq_x * random_values
    mixed_seq_y = shuffled_seq_y * random_values
    
    # 5. 加上原始序列
    mixed_seq_x += seq_x
    mixed_seq_y += seq_y
    
    return mixed_seq_x, mixed_seq_y

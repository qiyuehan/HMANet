from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.pre_train, args.label_len, args.pred_len],
        train_percent=args.train_percent,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader

import torch
from torchtext.data import Field, LabelField
from torchtext.data import Example
from torchtext.data import Dataset
from utils.util import *


def dataset_reader(train=True, fields=False):
    data_list1 = ['plat_form', 'biz_type', 'create_time', 'payed_time', 'cate1_id', 'cate2_id',
                  'preselling_shipped_time', 'seller_uid_field', 'company_name']
    data_and_label_list = ['lgst_company', 'warehouse_id', 'shipped_prov_id', 'shipped_city_id']
    data_list2 = ['rvcr_prov_name', 'rvcr_city_name']
    label_list = ['shipped_time', 'got_time', 'dlved_time', 'signed_time']
    signed_field = LabelField(sequential=False, batch_first=True, use_vocab=False, dtype=torch.double)
    label_field = LabelField(sequential=False, batch_first=True, use_vocab=False)
    if fields:
        regular_field = fields
    else:
        regular_field = Field(batch_first=True)

    field = []
    for d in data_list1:
        field.append((d, regular_field))
    for d in data_and_label_list:
        field.append((d, regular_field))
        field.append((d + '_label', label_field))
    for d in data_list2:
        field.append((d, regular_field))
    for d in label_list:
        field.append((d, signed_field))

    examples = []
    c = 0
    if train:
        path = r"SeedCup2019_pre/SeedCup_pre_train.csv"
        path_store = r"SeedCup2019_pre/data_train.txt"
    else:
        path = r"SeedCup2019_pre/SeedCup_pre_test.csv"
        path_store = r"SeedCup2019_pre/data_test.txt"

    process = False
    if process:
        if train:
            with open(path_store, "w+") as f:
                f.write('')
            with open(path) as f:
                _ = f.readline()
                line = f.readline()
                while line:
                    """
                    'uid', 'plat_form', 'biz_type', 'create_time', 'payed_time', 'product_id', 
                    'cate1_id', 'cate2_id', 'cate3_id', 'preselling_shipped_time', 'seller_uid', 
                    'company_name', 'lgst_company', 'warehouse_id', 'shipped_prov_id', 'shipped_city_id', 
                    'rvcr_prov_name', 'rvcr_city_name', 'shipped_time', 'got_time', 'dlved_time', 'signed_time'
                    """
                    tmp_list = []
                    start = 0
                    items = line.split('\t')
                    for i, data in enumerate(items):
                        # useless
                        if i in (0, 5, 8):
                            continue
                        # time
                        elif i in (18, 19, 20, 21):
                            if data == '-99':
                                tmp_list.append(float(-1))
                            else:
                                time = time_difference(start, data)
                                tmp_list.append(float(time))
                        elif i == 3:
                            start = data
                            tmp_list.append(start_difference(data) + '_' + str(i))
                        elif i == 4:
                            time = time_difference(start, data)
                            tmp_list.append(str(int(float(time))))
                        # pre sell
                        elif i == 9:
                            if data == '0':
                                tmp_list.append(data + '_' + 't')
                            else:
                                if 0 <= float(time_difference(start, data)) < 1000:
                                    tmp_list.append(str(int(float(time_difference(start, data)))))
                                else:
                                    tmp_list.append('0' + '_' + 't')
                        elif i in (12, 13, 14, 15):
                            tmp_list.append(data + '_' + str(i))
                            tmp_list.append(int(data))
                        else:
                            tmp_list.append(data + '_' + str(i))
                    with open(path_store, "a+") as txt:
                        context = ''
                        for i in tmp_list:
                            context += str(i) + '\t'
                        context = context[:-1] + '\n'
                        txt.write(context)
                    c += 1
                    print(c)
                    line = f.readline()
        else:
            with open(path_store, "w+") as f:
                f.write('')
            with open(path) as f:
                _ = f.readline()
                line = f.readline()
                while line:
                    """
                    'uid', 'plat_form', 'biz_type', 'create_time', 'payed_time', 'product_id', 
                    'cate1_id', 'cate2_id', 'cate3_id', 'preselling_shipped_time', 'seller_uid', 
                    'company_name', 'lgst_company', 'warehouse_id', 'shipped_prov_id', 'shipped_city_id', 
                    'rvcr_prov_name', 'rvcr_city_name', 'shipped_time', 'got_time', 'dlved_time', 'signed_time'
                    """
                    tmp_list = []
                    start = 0
                    items = line.split('\t')
                    items[-1] = items[-1][:-1]
                    for i, data in enumerate(items):
                        # useless
                        if i in (0, 5, 8):
                            continue
                        elif i == 3:
                            start = data
                            tmp_list.append(start_difference(data) + '_' + str(i))
                        elif i == 4:
                            time = time_difference(start, data)
                            tmp_list.append(str(int(float(time))))
                        # pre sell
                        elif i == 9:
                            if data == '0':
                                tmp_list.append(data + '_' + 't')
                            else:
                                if 0 <= float(time_difference(start, data)) < 1000:
                                    tmp_list.append(str(int(float(time_difference(start, data)))))
                                else:
                                    tmp_list.append('0_t')
                        elif i == 12:
                            for k in range(8):
                                tmp_list.append('0')
                            tmp_list.append(data + '_' + str(i + 4))
                        elif i == 13:
                            tmp_list.append(data + '_' + str(i + 4))
                        else:
                            tmp_list.append(data + '_' + str(i))
                    for k in range(4):
                        tmp_list.append('0')
                    with open(path_store, "a+") as txt:
                        context = ''
                        for i in tmp_list:
                            context += str(i) + '\t'
                        context = context[:-1] + '\n'
                        txt.write(context)
                    c += 1
                    print(c)
                    line = f.readline()

    if train:
        with open(path_store, "r") as f:
            line = f.readline()
            while line:
                items = line.split('\t')
                for i in (10, 12, 14, 16):
                    items[i] = int(items[i])
                for i in (19, 20, 21, 22):
                    items[i] = float(items[i])
                examples.append(Example.fromlist(items, field))
                line = f.readline()
    else:
        with open(path_store, "r") as f:
            line = f.readline()
            while line:
                items = line.split('\t')
                examples.append(Example.fromlist(items, field))
                line = f.readline()

    if train:
        length = examples.__len__()
        train_examples = Dataset(examples[:int(0.8 * length)], field)
        test_examples = Dataset(examples[int(0.8 * length):], field)
        return train_examples, test_examples, regular_field
    else:
        return Dataset(examples, field), regular_field


if __name__ == '__main__':
    dataset_reader()

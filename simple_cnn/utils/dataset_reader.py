import torch
from torchtext.data import Field, LabelField
from torchtext.data import Example
from torchtext.data import Dataset
from utils.util import *


def dataset_reader(train=True, fields=False, process=False):

    label_field = LabelField(sequential=False, batch_first=True, use_vocab=False,
                             dtype=torch.double)
    if fields:
        regular_field = fields
    else:
        regular_field = Field(batch_first=True)

    #
    train_data_list = ['plat_form', 'biz_type', 'create_time', 'create_hour',
                       'payed_day', 'payed_hour', 'cate1_id', 'cate2_id',
                       'cate3_id', 'preselling_shipped_day',
                       'preselling_shipped_hour', 'seller_uid_field',
                       'company_name', 'lgst_company', 'warehouse_id',
                       'shipped_prov_id', 'shipped_city_id', 'rvcr_prov_name',
                       'rvcr_city_name', 'shipped_day', 'shipped_hour',
                       'got_day', 'got_hour', 'dlved_day', 'dlved_hour']
    label_list = ['signed_day', 'signed_hour', 'lgst_company_label',
                  'warehouse_id_label', 'shipped_prov_id_label',
                  'shipped_city_id_label', 'shipped_day_label',
                  'shipped_hour_label', 'got_day_label', 'got_hour_label',
                  'dlved_day_label', 'dlved_hour_label']

    field = []
    for d in train_data_list:
        field.append((d, regular_field))
    for d in label_list:
        field.append((d, label_field))

    if train:
        path = r"SeedCup2019_pre/SeedCup_pre_train.csv"
        path_store = r"SeedCup2019_pre/data_train.txt"
    else:
        path = r"SeedCup2019_pre/SeedCup_pre_test.csv"
        path_store = r"SeedCup2019_pre/data_test.txt"

    if process:
        process_data(train, path, path_store)

    examples = []
    with open(path_store, "r") as f:
        line = f.readline()
        while line:
            items = list(line.split(' '))
            for i in (25, 26, 31, 32, 33, 34, 35, 36):
                items[i] = float(items[i])
            for i in (27, 28, 29, 30):
                items[i] = int(items[i])

            examples.append(Example.fromlist(items, field))
            line = f.readline()

    if train:
        length = examples.__len__()
        train_examples = Dataset(examples[:int(0.8 * length)], field)
        test_examples = Dataset(examples[int(0.8 * length):], field)
        return train_examples, test_examples, regular_field
    else:
        return Dataset(examples, field), regular_field


def process_data(train, path, path_store):
    c = 0

    with open(path_store, "w+") as f:
        f.write('')
    with open(path) as f:
        _ = f.readline()
        line = f.readline()
        while line:
            tmp_list = []
            start = 0
            items = line.split('\t')
            for i, data in enumerate(items):
                """
                'uid', 'plat_form', 'biz_type', 'create_time', 'payed_time', 
                'product_id', 'cate1_id', 'cate2_id', 'cate3_id', 
                'preselling_shipped_time', 'seller_uid', 'company_name', 
                'lgst_company', 'warehouse_id', 'shipped_prov_id', 'shipped_city_id', 
                'rvcr_prov_name', 'rvcr_city_name', 'shipped_time', 'got_time', 
                'dlved_time', 'signed_time'
                """

                # skip hidden data in test
                if i == 12 and not train:
                    # 4 hidden data
                    for j in range(4):
                        tmp_list.append(str(-99))
                    # rvcr prov, rvcr city
                    tmp_list.append(items[12] + '_16')
                    tmp_list.append(items[13][:-1] + '_17')
                    # hidden time
                    for j in range(18):
                        tmp_list.append(str(-99))
                    break

                # useless information
                if i in (0, 5):
                    continue

                # time
                elif i in (3, 4, 9, 18, 19, 20, 21):
                    # start day
                    if i == 3:
                        start_date = get_day(data)
                        tmp_list.append(start_date + '_' + str(start))
                        tmp_list.append(get_hour(data))
                    # pre sell time
                    elif i == 9:
                        if data != '0' and 0 < int(start_difference(data)) < 1000:
                            tmp_list.append(day_difference(start_date, get_day(data)))
                            tmp_list.append(get_hour(data))
                        else:
                            tmp_list.append(str(-99))
                            tmp_list.append(str(-99))
                    # noise time
                    elif data == '-99':
                        tmp_list.append(str(-99))
                        tmp_list.append(str(-99))
                    # pay, shipped, got, dlved, signed
                    else:
                        tmp_list.append(day_difference(start_date, get_day(data)))
                        tmp_list.append(get_hour(data))

                # hidden data
                elif i in (12, 13, 14, 15):
                    tmp_list.append(data + '_' + str(i))

                else:
                    tmp_list.append(data + '_' + str(i))

            # hidden data label
            if train:
                for i in (12, 13, 14, 15):
                    tmp_list.append(int(items[i]))
                for i in (18, 19, 20):
                    if items[i] == '-99':
                        tmp_list.append(str(-99))
                        tmp_list.append(str(-99))
                    else:
                        tmp_list.append(int(day_difference(start_date, get_day(items[i]))))
                        tmp_list.append(int(get_hour(items[i])))

            with open(path_store, "a+") as txt:
                context = ''
                for i in tmp_list:
                    context += str(i) + ' '
                context = context[:-1] + '\n'
                txt.write(context)

            c += 1
            print(c)
            line = f.readline()


if __name__ == '__main__':
    dataset_reader()

import os
from torchtext.data import BucketIterator

from modules.half_pred_cnn import FirstCNN
from utils import *


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test, field = dataset_reader(train=True, process=False)
    evl, _ = dataset_reader(train=False, fields=field, process=False)
    field.build_vocab(train, evl)
    evl_iter, test_iter = BucketIterator.splits(
        (evl, test),
        batch_sizes=(512, 512),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False,
        shuffle=False
        )
    c = 0
    model = FirstCNN(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    model.load_state_dict(torch.load('model/model.pkl'))
    data_path = r"data/data_test.txt"
    store_path = r"data/data_tmp1.txt"
    new_path = r"data/data_tmp2.txt"
    with open(store_path, "w+") as f:
        f.write('')
    with open(new_path, "w+") as f:
        f.write('')

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(evl_iter):
            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time, data.create_hour,
                                data.payed_day, data.payed_hour, data.cate1_id, data.cate2_id,
                                data.cate3_id, data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name,
                                data.rvcr_prov_name, data.rvcr_city_name), dim=1)
            prov, city, lgst, warehouse = model(inputs, field, train=True)
            prov = torch.argmax(prov, dim=1)
            city = torch.argmax(city, dim=1)
            lgst = torch.argmax(lgst, dim=1)
            warehouse = torch.argmax(warehouse, dim=1)
            for b in range(prov.size(0)):
                with open(store_path, "a+") as f:
                    f.write(str(int(lgst[b])) + '_12' + '\t' +
                            str(int(warehouse[b])) + '_13' + '\t' +
                            str(int(prov[b])) + '_14' + '\t' +
                            str(int(city[b])) + '_15' + '\n')
                    c += 1
                    print(0, c)

    c = 0
    with open(data_path, "r") as f1:
        with open(store_path, "r") as f2:
            with open(new_path, "w+") as f3:
                line1 = f1.readline()
                line2 = f2.readline()
                while line2:
                    items1 = line1.split(' ')
                    items2 = line2.split('\t')
                    items1[13] = items2[0]
                    items1[14] = items2[1]
                    items1[15] = items2[2]
                    items1[16] = items2[3][:-1]
                    w = ''
                    for i in items1:
                        w += i + ' '
                    f3.write(w[:-2] + '\n')
                    c += 1
                    print(1, c)
                    line1 = f1.readline()
                    line2 = f2.readline()

    os.remove(store_path)
    os.remove(data_path)
    os.rename(new_path, data_path)


if __name__ == '__main__':
    main()

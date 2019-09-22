from utils import *
from modules import *
import torch
from torch import nn
from torch import optim
from torchtext.data import BucketIterator
import arrow

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train, test, field = dataset_reader(train=True)
    evl, _ = dataset_reader(train=False, fields=field)
    field.build_vocab(train, evl)
    del evl
    train_iter, test_iter = BucketIterator.splits(
        (train, test),
        batch_sizes=(256, 256),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False,
        shuffle=True
        )

    model = Simple(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    criterion = RMSELoss(early=1, late=1.5)
    optimizer = optim.Adam((model.parameters()), lr=0.003, weight_decay=0.03)

    best = 9999
    loss_train = 0
    count_train = 0

    for epoch in range(10000):
        for i, data in enumerate(train_iter):
            # inputs = torch.cat((data.plat_form, data.biz_type, data.create_time, data.payed_time,
            #                     data.cate1_id, data.cate2_id, data.preselling_shipped_time,
            #                     data.seller_uid_field, data.company_name, data.rvcr_prov_name,
            #                     data.rvcr_city_name, data.lgst_company, data.warehouse_id,
            #                     data.shipped_prov_id, data.shipped_city_id,
            #                     data.shipped_time, data.got_time, data.dlved_time), dim=1)
            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time, data.payed_time,
                                data.cate1_id, data.cate2_id, data.preselling_shipped_time,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)
            t = model(inputs, 'train', field)

            loss = criterion(t * 200 + 50, data.signed_time.unsqueeze(1), train=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()
            count_train += 1

            if (i + 1) % 200 == 0:
                model.eval()
                with torch.no_grad():
                    loss_test = 0
                    count_test = 0
                    acc_count = 0
                    acc_total = 0
                    for j, data_t in enumerate(test_iter):
                        inputs = torch.cat((data_t.plat_form, data_t.biz_type, data_t.create_time, data_t.payed_time,
                                            data_t.cate1_id, data_t.cate2_id, data_t.preselling_shipped_time,
                                            data_t.seller_uid_field, data_t.company_name, data_t.rvcr_prov_name,
                                            data_t.rvcr_city_name), dim=1)
                        t = model(inputs, 'test', field)
                        loss = criterion((t * 200 + 50), data_t.signed_time.unsqueeze(1), train=False)
                        loss_test += loss.item()
                        count_test += 1

                        # get time acc
                        for b in range(t.size(0)):
                            start = arrow.get('2019-03-01 00:00:00').timestamp
                            create_time = field.vocab.itos[data.create_time[b, 0]].split('_')[0]
                            signed_time = data.signed_time[b]
                            signed_time = arrow.get(float(start) + float(signed_time) * 3600)
                            pred_time = float(start) + (float(create_time) + float(t[b, 0]) * 200 + 50) * 3600
                            pred_time = str(arrow.get(pred_time)).split('-')[2][:2]
                            signed_time = str(signed_time).split('-')[2][:2]
                            acc_total += 1
                            if int(signed_time) <= int(pred_time):
                                acc_count += 1

                    print('Epoch: %4d | Iter: %4d / %4d | Loss: %4.4f | Rank: %4.4f | '
                          'Time: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                     (loss_train / count_train) ** 0.5,
                                                     (loss_test / count_test) ** 0.5,
                                                     acc_count / acc_total,
                                                     'YES' if loss_test / count_test < best else 'NO'))
                    if loss_test / count_test < best:
                        best = loss_test / count_test
                        torch.save(model.state_dict(), r'model/model_' + str(int(best)) + r'.pkl')
                    count_train = 0
                    loss_train = 0
                model.train()


if __name__ == '__main__':
    main()

from utils import *
from modules import *
import torch
from torch import optim
from torchtext.data import BucketIterator

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train, test, field = dataset_reader(train=True, process=False)
    evl, _ = dataset_reader(train=False, fields=field, process=False)
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
    criterion_day = RMSELoss(gap=0, early=1, late=4)
    criterion_hour = RMSELoss(gap=0, early=1, late=1)
    optimizer = optim.Adam((model.parameters()), lr=0.0001, weight_decay=0.05)

    best = 99
    loss_train = 0
    count_train = 0

    for epoch in range(50):
        for i, data in enumerate(train_iter):

            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time,
                                data.create_hour, data.payed_day, data.payed_hour,
                                data.cate1_id, data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)
            day, hour = model(inputs, 'train', field)

            loss = (criterion_day(day * 8 + 4, data.signed_day.unsqueeze(1), train=True) +
                    criterion_hour(hour * 10 + 15, data.signed_hour.unsqueeze(1), train=True))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()
            count_train += 1

            if (i + 1) % 300 == 0:
                model.eval()
                with torch.no_grad():
                    loss_test = 0
                    count_test = 0
                    acc_count = 0
                    acc_total = 0
                    for j, data_t in enumerate(test_iter):
                        inputs = torch.cat((data_t.plat_form, data_t.biz_type, data_t.create_time,
                                            data_t.create_hour, data_t.payed_day, data_t.payed_hour,
                                            data_t.cate1_id, data_t.cate2_id, data_t.cate3_id,
                                            data_t.preselling_shipped_day, data_t.preselling_shipped_hour,
                                            data_t.seller_uid_field, data_t.company_name, data_t.rvcr_prov_name,
                                            data_t.rvcr_city_name), dim=1)
                        day, hour = model(inputs, 'test', field)

                        loss = (criterion_day(day * 8 + 4, data_t.signed_day.unsqueeze(1), train=False) +
                                criterion_hour(hour * 10 + 15, data_t.signed_hour.unsqueeze(1), train=False))
                        loss_test += loss.item()
                        count_test += 1

                        # get time acc
                        for b in range(day.size(0)):
                            acc_total += 1
                            if int(day[b] * 8 + 4) <= int(data_t.signed_day[b]):
                                acc_count += 1

                    print('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | Rank: %.3f | '
                          'Time: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                     loss_train / count_train,
                                                     loss_test / count_test,
                                                     acc_count / acc_total,
                                                     ('YES' if loss_test / count_test < best and
                                                     acc_count / acc_total >= 0.981 else 'NO')))
                    if loss_test / count_test < best and acc_count / acc_total >= 0.981:
                        best = loss_test / count_test
                        torch.save(model.state_dict(), r'model/model_' + str(int(best)) + r'.pkl')
                    count_train = 0
                    loss_train = 0
                model.train()


if __name__ == '__main__':
    main()

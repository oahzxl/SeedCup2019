import torch
from torch import optim
from torchtext.data import BucketIterator

from modules import *
from utils import *

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

    model_imate = Simple(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    optimizer_imate = optim.Adam((model_imate.parameters()), lr=0.0001, weight_decay=0.1)

    model = Simple(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    optimizer = optim.Adam((model.parameters()), lr=0.0001, weight_decay=0.1)

    criterion_day = RMSELoss(gap=0, early=1, late=8)
    criterion_hour = RMSELoss(gap=0, early=2, late=2)

    best = 99
    train_loss = 0
    train_count = 0

    for epoch in range(20):
        for i, data in enumerate(train_iter):

            inputs_imate = torch.cat((data.plat_form, data.biz_type, data.create_time,
                                data.create_hour, data.payed_day, data.payed_hour,
                                data.cate1_id, data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)
            day_imate, hour_imate = model_imate(inputs_imate, 'train', field)

            loss_imate = (criterion_day(day * 8 + 3, data.signed_day.unsqueeze(1), train=True) +
                    criterion_hour(hour * 10 + 15, data.signed_hour.unsqueeze(1), train=True))
            loss_imate.backward()
            optimizer_imate.step()
            optimizer_imate.zero_grad()

            train_loss += loss_imate.item()
            train_count += 1

            if (i + 1) % 300 == 0:
                model.eval()
                with torch.no_grad():
                    rank = 0
                    acc = 0
                    count = 0
                    for j, data_t in enumerate(test_iter):
                        if j > 50:
                            break

                        inputs = torch.cat((data_t.plat_form, data_t.biz_type, data_t.create_time,
                                            data_t.create_hour, data_t.payed_day, data_t.payed_hour,
                                            data_t.cate1_id, data_t.cate2_id, data_t.cate3_id,
                                            data_t.preselling_shipped_day, data_t.preselling_shipped_hour,
                                            data_t.seller_uid_field, data_t.company_name, data_t.rvcr_prov_name,
                                            data_t.rvcr_city_name), dim=1)
                        day, hour = model(inputs, 'test', field)

                        for b in range(day.size(0)):
                            # time
                            if int('%.0f' % (day[b] * 8 + 3)) <= int(data_t.signed_day[b]):
                                acc += 1

                            # rank
                            pred_time = arrow.get("2019-03-" + ('%.0f' % (day[b] * 8 + 3 + 3)).zfill(2) +
                                                  ' ' + ('%.0f' % (hour[b] * 10 + 15)).zfill(2))
                            sign_time = arrow.get("2019-03-" + str(int(data_t.signed_day[b]) + 3).zfill(2) + ' ' +
                                                  str(int(data_t.signed_hour[b])).zfill(2))
                            rank += int((pred_time - sign_time).seconds / 3600) ** 2

                            count += 1

                    acc = acc / count
                    rank = (rank / count) ** 0.5

                    print('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | Rank: %.3f | '
                          'Time: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                     train_loss / train_count, rank, acc,
                                                     ('YES' if rank < best and acc >= 0.981 else 'NO')))
                    if rank < best and acc >= 0.981:
                        best = rank
                        torch.save(model_imate.state_dict(), r'model/model_' + str(int(best)) + r'.pkl')

                    train_count = 0
                    train_loss = 0
                model.train()


if __name__ == '__main__':
    main()

from torch import optim
from torchtext.data import BucketIterator

from modules import *
from utils import *


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test, field = dataset_reader(train=True, process=False)
    evl, _ = dataset_reader(train=False, fields=field, process=False)
    field.build_vocab(train, evl)
    del evl
    train_iter, test_iter = BucketIterator.splits(
        (train, test),
        batch_sizes=(512, 512),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False,
        shuffle=True
        )

    model = SimpleRNN(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    criterion_day = RMSELoss(gap=0, early=1, late=3)
    criterion_last_day = RMSELoss(gap=0, early=2, late=7)
    criterion_hour = RMSELoss(gap=0, early=1, late=1)
    optimizer = optim.Adam((model.parameters()), lr=0.00003, weight_decay=0.0)
    optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=False,
                                         threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    with open(r"model/simple_rnn_log.txt", "w+") as f:
        f.write('')
    best = 99
    train_loss = 0
    train_count = 0

    for epoch in range(50):
        for i, data in enumerate(train_iter):

            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time,
                                data.create_hour, data.payed_day, data.payed_hour,
                                data.cate1_id, data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)

            outputs = model(inputs, 'train', field)
            # loss = (criterion_hour(outputs[0] + 0.5, data.shipped_day_label.unsqueeze(1), train=True) +
            #         criterion_hour(outputs[1] * 5 + 15, data.shipped_hour_label.unsqueeze(1), train=True) +
            #         criterion_hour(outputs[0] + 0.5 + outputs[2] + 0.4, data.got_day_label.unsqueeze(1), train=True) +
            #         criterion_hour(outputs[3] * 5 + 15, data.got_hour_label.unsqueeze(1), train=True) +
            #         criterion_hour(outputs[0] + 0.5 + outputs[2] + 0.4 + outputs[4] + 0.4,
            #                        data.dlved_day_label.unsqueeze(1), train=True) +
            #         criterion_hour(outputs[5] * 5 + 15, data.dlved_hour_label.unsqueeze(1), train=True) +
            #         criterion_day(outputs[0] + 0.5 + outputs[2] + 0.4 + outputs[4] + 0.4 + outputs[6] * 2 + 1,
            #                       data.signed_day.unsqueeze(1), train=True) +
            #         criterion_hour(outputs[7] * 5 + 15, data.signed_hour.unsqueeze(1), train=True)
            #         )
            loss = (criterion_day(outputs[0] * 2 + 1, data.shipped_day_label.unsqueeze(1), train=True) +
                    criterion_hour(outputs[1] * 5 + 15, data.shipped_hour_label.unsqueeze(1), train=True) +
                    criterion_day(outputs[2] * 2 + 1, data.got_day_label.unsqueeze(1), train=True) +
                    criterion_hour(outputs[3] * 5 + 15, data.got_hour_label.unsqueeze(1), train=True) +
                    criterion_day(outputs[4] * 2 + 1, data.dlved_day_label.unsqueeze(1), train=True) +
                    criterion_hour(outputs[5] * 5 + 15, data.dlved_hour_label.unsqueeze(1), train=True) +
                    criterion_last_day(outputs[6] * 3 + 3, data.signed_day.unsqueeze(1), train=True) +
                    criterion_hour(outputs[7] * 5 + 15, data.signed_hour.unsqueeze(1), train=True)
                    )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_count += 1

            if (i + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    rank = 0
                    acc = 0
                    count = 0
                    for j, data_t in enumerate(test_iter):
                        if j > 5:
                            break

                        inputs = torch.cat((data_t.plat_form, data_t.biz_type, data_t.create_time,
                                            data_t.create_hour, data_t.payed_day, data_t.payed_hour,
                                            data_t.cate1_id, data_t.cate2_id, data_t.cate3_id,
                                            data_t.preselling_shipped_day, data_t.preselling_shipped_hour,
                                            data_t.seller_uid_field, data_t.company_name, data_t.rvcr_prov_name,
                                            data_t.rvcr_city_name), dim=1)
                        outputs = model(inputs, 'test', field)
                        # day = outputs[0] + 0.5 + outputs[2] + 0.4 + outputs[4] + 0.4 + outputs[6] * 2 + 1
                        day = outputs[6] * 3 + 3
                        hour = outputs[-1]

                        for b in range(day.size(0)):

                            # rank
                            if int(data_t.signed_day[b]) < 0 or int(data_t.signed_day[b]) > 20:
                                continue
                            pred_time = arrow.get("2019-03-" + ('%.0f' % (day[b] + 3)).zfill(2) +
                                                  ' ' + ('%.0f' % (hour[b] * 5 + 15)).zfill(2))
                            sign_time = arrow.get("2019-03-" + str(int(data_t.signed_day[b]) + 3).zfill(2) + ' ' +
                                                  str(int(data_t.signed_hour[b])).zfill(2))
                            rank += int((pred_time.timestamp - sign_time.timestamp) / 3600) ** 2

                            # time
                            if int('%.0f' % day[b]) <= int(data_t.signed_day[b]):
                                acc += 1

                            count += 1

                    acc = acc / count
                    rank = (rank / count) ** 0.5

                    print('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | Rank: %.3f | '
                          'Time: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                     train_loss / train_count, rank, acc,
                                                     ('YES' if rank < best and acc >= 0.981 else 'NO')))
                    with open(r"model/simple_rnn_log.txt", "a+") as f:
                        f.write('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | Rank: %.3f | '
                                'Time: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                           train_loss / train_count, rank, acc,
                                                           ('YES' if rank < best and acc >= 0.981 else 'NO')))
                    if rank < best and acc >= 0.981:
                        best = rank
                        torch.save(model.state_dict(), r'model/simple_rnn_model_' + str(int(best)) + r'.pkl')

                    train_count = 0
                    train_loss = 0
                model.train()


if __name__ == '__main__':
    main()

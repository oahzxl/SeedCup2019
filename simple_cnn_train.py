import argparse

from torch import optim
from torchtext.data import BucketIterator

from modules import *
from utils import *


parser = argparse.ArgumentParser(description='RNN + CNN')
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.00003, help='initial learning rate [default: 0.0003]')
learn.add_argument('--late', type=float, default=7, help='punishment of delay [default: 8]')
learn.add_argument('--batch_size', type=int, default=1024, help='batch size for training [default: 1024]')
learn.add_argument('--checkpoint', type=str, default='N', help='load latest model [default: N]')
learn.add_argument('--process', type=str, default='N', help='preprocess data [default: N]')
learn.add_argument('--interval', type=int, default=300, help='test interval [default: 300]')


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.process == 'Y':
        train, test, field = dataset_reader(train=True, process=True, stop=1200000)
        evl, _ = dataset_reader(train=False, fields=field, process=True)
    else:
        train, test, field = dataset_reader(train=True, process=False, stop=6000)
        evl, _ = dataset_reader(train=False, fields=field, process=False, stop=6)

    field.build_vocab(train, evl)
    del evl
    train_iter, test_iter = BucketIterator.splits(
        (train, test),
        batch_sizes=(args.batch_size, args.batch_size),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False,
        shuffle=True
    )

    model = SimpleCNN(num_embeddings=len(field.vocab), embedding_dim=16).to(device)
    criterion_last_day = RMSELoss(gap=0, early=1, late=args.late)
    optimizer = optim.Adam((model.parameters()), lr=args.lr, weight_decay=0.03)
    with open(r"model/simple_cnn_log.txt", "w+") as f:
        f.write('')
    if args.checkpoint == 'Y':
        model.load_state_dict(torch.load('model/simple_cnn_model.pkl'))

    best = 99
    train_loss = 0
    train_count = 0
    for epoch in range(200):
        for i, data in enumerate(train_iter):

            inputs = torch.cat((data.plat_form, data.biz_type,
                                data.payed_day, data.payed_hour,
                                data.cate1_id, data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day,
                                data.seller_uid_field, data.company_name,
                                data.rvcr_prov_name, data.rvcr_city_name,
                                ), dim=1)

            outputs = model(inputs, 'train', field)
            loss = criterion_last_day(outputs * 3 + 3, data.signed_day.unsqueeze(1), train=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_count += 1

            if (i + 1) % args.interval == 0:
                model.eval()
                with torch.no_grad():
                    rank = 0
                    acc = 0
                    count = 0
                    test_loss = 0
                    for j, data_t in enumerate(test_iter):
                        if j > (args.interval / 2):
                            break

                        inputs = torch.cat((data_t.plat_form, data_t.biz_type,
                                            data_t.payed_day, data_t.payed_hour,
                                            data_t.cate1_id, data_t.cate2_id, data_t.cate3_id,
                                            data_t.preselling_shipped_day,
                                            data_t.seller_uid_field, data_t.company_name,
                                            data_t.rvcr_prov_name, data_t.rvcr_city_name,
                                            ), dim=1)

                        outputs = model(inputs, 'test', field)

                        loss = criterion_last_day(outputs * 3 + 3, data_t.signed_day.unsqueeze(1), train=True)
                        test_loss += loss.item()

                        day = outputs * 3 + 3

                        for b in range(day.size(0)):

                            # rank
                            if not (0 <= int(data_t.signed_day[b]) <= 25) or not (0 <= day[b] <= 25):
                                continue
                            pred_time = arrow.get("2019-03-" + ('%.0f' % (day[b] + 5)).zfill(2) +
                                                  ' 15')
                            sign_time = arrow.get("2019-03-" + str(int(data_t.signed_day[b]) + 5).zfill(2) + ' ' +
                                                  str(int(data_t.signed_hour[b])).zfill(2))
                            rank += int((pred_time.timestamp - sign_time.timestamp) / 3600) ** 2

                            # time
                            if int('%.0f' % day[b]) <= int(data_t.signed_day[b]):
                                acc += 1

                            count += 1

                    acc = acc / count
                    test_loss = test_loss / count
                    rank = (rank / count) ** 0.5

                    print('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | Test Loss: %.3f | Rank: %.3f | '
                          'Time: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                     train_loss / train_count, test_loss * day.size(0), rank, acc,
                                                     ('YES' if rank < best and acc >= 0.981 else 'NO')))
                    with open(r"model/simple_cnn_log.txt", "a+") as f:
                        f.write('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | Test Loss: %.3f | Rank: %.3f | '
                                'Time: %.3f | Best: %s\n' % (epoch, (i + 1), train_iter.__len__(),
                                                             train_loss / train_count, test_loss * day.size(0),
                                                             rank, acc,
                                                             ('YES' if rank < best and acc >= 0.981 else 'NO')))
                    if rank < best and acc >= 0.981:
                        best = rank
                        torch.save(model.state_dict(), r'model/simple_cnn_model_' + str(int(best)) + '.pkl')

                    train_count = 0
                    train_loss = 0
                model.train()


if __name__ == '__main__':
    main()
from torch import nn
from torch import optim
from torchtext.data import BucketIterator
from utils import *
from modules.first_cnn import FirstCNN


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test, field = dataset_reader(train=True, process=False)
    evl, _ = dataset_reader(train=False, fields=field, process=False)
    field.build_vocab(train, evl)
    del evl
    train_iter, test_iter = BucketIterator.splits(
        (train, test),
        batch_sizes=(128, 128),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False,
        shuffle=True
        )

    model = FirstCNN(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam((model.parameters()), lr=0.001, weight_decay=0.1)
    optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=False,
                                         threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    best = 0
    train_loss = 0
    train_count = 0

    for epoch in range(100):
        for i, data in enumerate(train_iter):

            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time, data.create_hour,
                                data.payed_day, data.payed_hour, data.cate1_id, data.cate2_id,
                                data.cate3_id, data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name,
                                data.rvcr_prov_name, data.rvcr_city_name), dim=1)
            prov, city, lgst, warehouse = model(inputs, field, train=True)

            loss = (criterion_ce(prov, data.shipped_prov_id_label.long()) +
                    criterion_ce(city, data.shipped_city_id_label.long()) +
                    criterion_ce(lgst, data.lgst_company_label.long()) +
                    criterion_ce(warehouse, data.warehouse_id_label.long()))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_count += 1

            if (i + 1) % 300 == 0:
                model.eval()
                with torch.no_grad():
                    acc = 0
                    count = 0
                    for j, data_t in enumerate(test_iter):
                        if j > 5:
                            break

                        inputs = torch.cat((data_t.plat_form, data_t.biz_type, data_t.create_time, data_t.create_hour,
                                            data_t.payed_day, data_t.payed_hour, data_t.cate1_id, data_t.cate2_id,
                                            data_t.cate3_id, data_t.preselling_shipped_day,
                                            data_t.preselling_shipped_hour, data_t.seller_uid_field,
                                            data_t.company_name, data_t.rvcr_prov_name, data_t.rvcr_city_name), dim=1)

                        prov, city, lgst, warehouse = model(inputs, field, train=False)

                        for d, l in [(prov, data_t.shipped_prov_id_label), (city, data_t.shipped_city_id_label),
                                     (lgst, data_t.lgst_company_label), (warehouse, data_t.warehouse_id_label)]:
                            d = torch.argmax(d, dim=1)
                            acc += torch.sum(d == l.long())
                            count += d.size(0)

                    acc = float(acc) / count

                    print('Epoch: %3d | Iter: %4d / %4d | Loss: %.3f | '
                          'Acc: %.3f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                    train_loss / train_count, acc,
                                                    ('YES' if acc > best else 'NO')))
                    if acc > best:
                        best = acc
                        torch.save(model.state_dict(), r'model/model_' + str(int(best)) + r'.pkl')

                    train_count = 0
                    train_loss = 0
                model.train()


if __name__ == '__main__':
    main()

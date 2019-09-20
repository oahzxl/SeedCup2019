from utils import *
from modules import *
import torch
from torch import nn
from torch import optim
from torchtext.data import BucketIterator
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train, test, field = dataset_reader()
    field.build_vocab(train)
    train_iter, test_iter = BucketIterator.splits(
        (train, test),
        batch_sizes=(64, 64),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False
        )

    model = CNN(num_embeddings=len(field.vocab), embedding_dim=128).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam((model.parameters()), lr=0.03)
    best = 999
    loss_running = 0
    loss_count = 0
    for epoch in range(10000):
        for i, data in enumerate(train_iter):
            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time, data.payed_time,
                                data.cate1_id, data.cate2_id, data.preselling_shipped_time,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name, data.lgst_company, data.warehouse_id,
                                data.shipped_prov_id, data.shipped_city_id), dim=1)
            lgst, warehouse, prov, city, t1, t2, t3, t4 = model(inputs, 'train', field)

            loss = (criterion_ce(lgst, data.lgst_company_label) +
                    criterion_ce(warehouse, data.warehouse_id_label) +
                    criterion_ce(prov, data.shipped_prov_id_label) +
                    criterion_ce(city, data.shipped_city_id_label) +
                    criterion_mse(t1, (data.shipped_time.unsqueeze(1) - 100) / 300) +
                    criterion_mse(t2, (data.got_time.unsqueeze(1) - 100) / 300) +
                    criterion_mse(t3, (data.dlved_time.unsqueeze(1) - 100) / 300) +
                    criterion_mse(t4, (data.signed_time.unsqueeze(1) - 100) / 300))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_running += loss.item()
            loss_count += 1

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    loss_test = 0
                    test_count = 0
                    for j, data_t in enumerate(test_iter):
                        inputs = torch.cat((data_t.plat_form, data_t.biz_type, data_t.create_time, data_t.payed_time,
                                            data_t.cate1_id, data_t.cate2_id, data_t.preselling_shipped_time,
                                            data_t.seller_uid_field, data_t.company_name, data_t.rvcr_prov_name,
                                            data_t.rvcr_city_name), dim=1)
                        t = model(inputs, 'test', field)
                        loss = criterion_mse(t * 300 + 100, data_t.signed_time.unsqueeze(1))
                        loss_test += loss.item()
                        test_count += 1
                        if j > 3:
                            break
                    print('Epoch: %4d | Iter: %4d / %4d | Train Loss: %4.4f | '
                          'Test Loss: %4.4f | Best: %s' % (epoch, (i + 1), train_iter.__len__(),
                                                           loss_running / loss_count, loss_test / test_count,
                                                           'YES' if loss_test / test_iter.__len__() < best else 'NO'))
                    if loss_test / test_iter.__len__() < best:
                        torch.save(model.state_dict(), 'model' + str(int(loss_test / test_iter.__len__())) + '.pkl')
                        best = loss_test / test_iter.__len__()
                    loss_count = 0
                    loss_running = 0


if __name__ == '__main__':
    main()

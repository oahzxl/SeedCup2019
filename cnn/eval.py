from utils import *
from modules import *
import torch
import arrow
from torchtext.data import BucketIterator

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train, test, field = dataset_reader(train=True)
    evl, _ = dataset_reader(train=False, fields=field)
    field.build_vocab(train, evl)
    _, evl_iter = BucketIterator.splits(
        (train, evl),
        batch_sizes=(256, 256),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False
        )

    model = Simple(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    model.load_state_dict(torch.load('model/model.pkl'))
    with open('SeedCup2019_pre/result.txt', 'w+') as f:
        f.write('')

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(evl_iter):
            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time, data.payed_time,
                                data.cate1_id, data.cate2_id, data.preselling_shipped_time,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)
            t = model(inputs, 'test', field)
            with open('SeedCup2019_pre/result.txt', 'a+') as f:
                for b in range(t.size(0)):
                    start = arrow.get('2019-03-01 00:00:00').timestamp
                    create_time = field.vocab.itos[data.create_time[b, 0]].split('_')[0]
                    final = float(start) + 3600 * (float(create_time) + float(t[b, 0]) * 200 + 50)
                    final = str(arrow.get(final)).split('T')
                    final = final[0] + ' ' + final[1][:2]
                    f.write(final + '\n')


if __name__ == '__main__':
    main()

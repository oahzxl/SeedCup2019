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
            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time,
                                data.create_hour, data.payed_day, data.payed_hour,
                                data.cate1_id, data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)
            day, hour = model(inputs, 'test', field)
            with open('SeedCup2019_pre/result.txt', 'a+') as f:
                for b in range(day.size(0)):
                    final = '2019-03-' + ('%.0f' % (day * 8 + 4)).zfill(2) + ' ' + ('%.0f' % (hour * 10 + 15)).zfill(2)
                    f.write(final + '\n')


if __name__ == '__main__':
    main()

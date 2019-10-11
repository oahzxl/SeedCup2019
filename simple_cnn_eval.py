from torchtext.data import BucketIterator

from modules import *
from utils import *

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

    model = SimpleCNN(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    model.load_state_dict(torch.load('model/simple_cnn_model.pkl'))
    with open('data/simple_cnn_result.txt', 'w+') as f:
        f.write('')

    model.eval()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(evl_iter), total=evl_iter.__len__()):
            inputs = torch.cat((data.plat_form, data.biz_type, data.create_time,
                                data.create_hour, data.payed_day, data.payed_hour,
                                data.cate1_id, data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day, data.preselling_shipped_hour,
                                data.seller_uid_field, data.company_name, data.rvcr_prov_name,
                                data.rvcr_city_name), dim=1)
            day, hour = model(inputs, 'test', field)
            with open('data/simple_cnn_result.txt', 'a+') as f:
                for b in range(day.size(0)):
                    start_day = field.vocab.itos[data.create_time[b]][:2]
                    sign_day = int('%.0f' % (day[b] * 8 + 3)) + int(start_day)
                    sign_day = str(sign_day).zfill(2)
                    sign_hour = ('%.0f' % (hour[b] * 10 + 15)).zfill(2)
                    final = '2019-03-' + sign_day + ' ' + sign_hour
                    f.write(final + '\n')


if __name__ == '__main__':
    main()
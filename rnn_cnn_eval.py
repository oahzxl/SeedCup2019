from torchtext.data import BucketIterator

from modules import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train, test, field = dataset_reader(train=True, stop=900000)
    evl, _ = dataset_reader(train=False, fields=field)
    field.build_vocab(train, evl)
    _, evl_iter = BucketIterator.splits(
        (train, evl),
        batch_sizes=(1024, 1024),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False
        )

    model = RNNCNN(num_embeddings=len(field.vocab), embedding_dim=256).to(device)
    model.load_state_dict(torch.load('model/rnn_cnn_model.pkl'))
    with open('data/rnn_cnn_result.txt', 'w+') as f:
        f.write('')

    model.eval()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(evl_iter), total=evl_iter.__len__()):
            inputs = torch.cat((data.plat_form, data.biz_type,
                                data.payed_hour,
                                data.cate2_id, data.cate3_id,
                                data.preselling_shipped_day,
                                data.seller_uid_field, data.company_name,
                                data.lgst_company, data.warehouse_id,
                                data.rvcr_prov_name, data.rvcr_city_name,
                                data.shipped_prov_id, data.shipped_city_id,
                                ), dim=1)
            outputs = model(inputs, 'test', field)
            day = outputs * 3 + 3
            with open('data/rnn_cnn_result.txt', 'a+') as f:
                for b in range(day.size(0)):
                    start_day = '2019-' + field.vocab.itos[data.create_time[b]][:-2]
                    start_day = arrow.get(start_day).timestamp
                    sign_day = int('%.0f' % day[b])
                    sign_hour = '15'
                    final = str(arrow.get(start_day + sign_day * 24 * 60 * 60))[:10]
                    final = final + ' ' + sign_hour
                    f.write(final + '\n')


if __name__ == '__main__':
    main()

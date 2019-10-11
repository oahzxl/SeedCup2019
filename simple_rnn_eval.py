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
        batch_sizes=(512, 512),
        device=device,
        sort_within_batch=False,
        repeat=False,
        sort=False
        )

    model = SimpleRNN(num_embeddings=len(field.vocab), embedding_dim=300).to(device)
    model.load_state_dict(torch.load('model/simple_rnn_model.pkl'))
    with open('data/simple_rnn_result.txt', 'w+') as f:
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
            outputs = model(inputs, 'test', field)
            day = outputs[6] * 3 + 3
            hour = outputs[-1]
            with open('data/simple_rnn_result.txt', 'a+') as f:
                for b in range(day.size(0)):
                    start_day = arrow.get('2019-' + field.vocab.itos[data.create_time[b]]).timestamp
                    sign_day = int('%.0f' % day[b])
                    sign_hour = ('%.0f' % (hour[b] * 5 + 15)).zfill(2)
                    final = arrow.get(start_day + sign_day * 24 * 60 * 60) + sign_hour
                    f.write(final + '\n')


if __name__ == '__main__':
    main()

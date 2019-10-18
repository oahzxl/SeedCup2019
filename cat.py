from torchtext.data import BucketIterator

from modules import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train, test, field = dataset_reader(train=True, stop=1200000)
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

    with open('data/simple_rnn_result.txt', 'w+') as f:
        f.write('')

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(evl_iter), total=evl_iter.__len__()):

            day = outputs[6] * 3 + 3
            hour = outputs[-1]

            with open('data/simple_rnn_result.txt', 'a+') as f:
                for b in range(day.size(0)):
                    start_day = field.vocab.itos[data.create_time[b]][:-2]
                    start_day = arrow.get(start_day).timestamp
                    sign_day = int('%.0f' % day[b])
                    sign_hour = ('%.0f' % (hour[b] * 5 + 15)).zfill(2)
                    final = str(arrow.get(start_day + sign_day * 24 * 60 * 60))[:10]
                    final = final + ' ' + sign_hour
                    f.write(final + '\n')


if __name__ == '__main__':
    main()

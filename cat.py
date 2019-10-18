from torchtext.data import BucketIterator

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

    with open('data/result.txt', 'w+') as f:
        f.write('')

    with open('data/2.txt', "r") as f1:
        with open('data/3.txt', "r") as f2:
            for i, data in tqdm.tqdm(enumerate(evl_iter), total=evl_iter.__len__()):
                with open('data/result.txt', 'a+') as f:
                    for b in range(data.create_time.size(0)):
                        start_day = field.vocab.itos[data.create_time[b]][:-2]
                        start_day = arrow.get('2019-' + start_day).timestamp
                        day = (float(f1.readline()) + float(f2.readline())) / 2
                        sign_day = int('%.0f' % day)
                        sign_hour = '15'
                        final = str(arrow.get(start_day + sign_day * 24 * 60 * 60))[:10]
                        final = final + ' ' + sign_hour
                        f.write(final + '\n')


if __name__ == '__main__':
    main()

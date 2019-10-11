import arrow


def start_difference(time):
    time_standard = arrow.get('2019-03-01 00:00:00').timestamp
    t = arrow.get(time).timestamp
    return str(int((t - time_standard) / 3600))


def time_difference(time1, time2):
    time1 = arrow.get(time1).timestamp
    time2 = arrow.get(time2).timestamp
    return str(float((time2 - time1) / 3600))


def get_day(time):
    day = time[5:10]
    return day


def get_hour(time):
    hour = time[11:13]
    return hour


def day_difference(time1, time2):
    time1 = arrow.get(time1).timestamp
    time2 = arrow.get(time2).timestamp
    return str(int((time2 - time1) / 24 / 60 / 60))


def get_date(time):
    time = time[:10]
    return time


def start_difference(time):
    import arrow
    time_standard = arrow.get('2019-03-01 00:00:00').timestamp
    t = arrow.get(time).timestamp
    return str(int((t - time_standard) / 3600))


def time_difference(time1, time2):
    import arrow
    time1 = arrow.get(time1).timestamp
    time2 = arrow.get(time2).timestamp
    return str(float((time2 - time1) / 3600))

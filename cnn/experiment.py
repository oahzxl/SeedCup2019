import arrow
import datetime
from utils import *

path = r"../SeedCup2019_pre/data.txt"
m = 0
c = 0
# with open(path, 'r') as f:
#     _ = f.readline()
#     line = f.readline()
#     while line:
#         """
#         'uid', 'plat_form', 'biz_type', 'create_time', 'payed_time', 'product_id',
#         'cate1_id', 'cate2_id', 'cate3_id', 'preselling_shipped_time', 'seller_uid',
#         'company_name', 'lgst_company', 'warehouse_id', 'shipped_prov_id', 'shipped_city_id',
#         'rvcr_prov_name', 'rvcr_city_name', 'shipped_time', 'got_time', 'dlved_time', 'signed_time'
#         """
#         start = line.split('\t')[15]
#         if int(start) > m:
#             m = int(start)
#         c += 1
#         print(m)
#         line = f.readline()

# max 758
with open(path, 'r') as f:
    line = f.readline()
    while line:
        c += 1
        print(c)
        line = f.readline()
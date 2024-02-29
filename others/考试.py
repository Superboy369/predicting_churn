import pandas as pd

chipo = pd.read_csv('D:\\@我的记录文件夹\\生产实习python深度学习\\chipotle.tsv',encoding = 'ANSI')
min_sum_item = chipo.groupby('item_name')['quantity'].sum().min()
# count = chipo.nunique('item_name')
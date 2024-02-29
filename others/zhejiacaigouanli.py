import pandas as pd

caigouruku = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\data1\\家电采购入库浙江.csv',encoding = 'ANSI')
jiuyin = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\data1\\浙江酒饮.csv',encoding = 'ANSI')
muyingruku = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\data1\\浙江母婴入库.csv',encoding = 'ANSI')
rihua = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\data1\\浙江日化.csv',encoding = 'ANSI')
shipinruku = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\data1\\浙江食品入库.csv',encoding = 'ANSI')
print(rihua,rihua.shape)

caigouruku['行业'] = '家电行业'
jiuyin['行业'] = '酒饮行业'
muyingruku['行业'] = '母婴行业'
rihua['行业'] = '日化行业'
shipinruku['行业'] = '食品行业'

total_data = caigouruku.append(jiuyin)
total_data = total_data.append(muyingruku)
total_data = total_data.append(rihua)
total_data = total_data.append(shipinruku)

total_data.columns

total_data[['平台','项目','商品名称','商品指导采购价','商品品牌','商品税收名称','行业','采购额']]

total_data.isnull().sum()

sum_null = total_data.loc[total_data.商品税收名称.isnull(),'采购额'].sum() # 统计缺失数据采购额总和
sum_baby = total_data.loc[total_data['行业'] == '母婴行业','采购额'].sum() # 统计母婴行业采购额总和
sum_null / sum_baby

all_data = total_data[-(total_data['商品税收名称'].isnull())] # 删除缺失值

all_data.行业.value_counts()

print(type(all_data['行业'] == '家电行业'))
print(all_data.loc[all_data['行业'] == '家电行业','商品税收名称'].unique())

# 找到家电行业中属于酒饮行业的数据，替换行业为”酒饮行业“
all_data.loc[(all_data['行业'] == '家电行业') & (all_data['商品税收名称'].str.contains('白酒|葡萄酒')),'行业'] = '酒饮行业'
# 找到家电行业中属于日化行业的数据，替换行业为“日化行业”
all_data.loc[(all_data['行业'] == '家电行业') & (all_data['商品税收名称'].str.contains('合成洗涤剂|纯碱类|其他无机酸|金属硝酸盐、亚硝酸盐|烧碱|特种纤维及高功能化工产品|有机－无机化合物|其他未列明纺织产品|金属制日用杂品|卫生用纸制品|其他金属工艺品|卫生材料及敷料|其他服装|磷化物、金属磷酸盐|其他体育用品|工艺陶瓷制品|消毒防腐及创伤外科用药| 床褥单|其他口腔清洁护理用品|清洁类化妆品|氰化物、氧氰化物及氰络合物|无环烃饱和氯化衍生物')),'行业']='日化行业'
# 找到食品行业中属于酒饮行业的数据，替换行业为“酒饮行业”
all_data.loc[(all_data['行业'] == '食品行业') & (all_data['商品税收名称'].str.contains('果汁和蔬菜汁类饮料|其他软饮料|蛋白饮料|固体饮料|茶饮料|药酒|包装饮用水')),'行业'] = '酒饮行业'
# 找到食品行业中包含母婴行业相关字符数据，替换行业为“酒饮行业”
all_data.loc[all_data['商品税收名称'].str.contains('儿童|婴儿|奶嘴|母乳|婴幼儿|月子|医疗、卫生用橡胶制品'),'行业'] = '母婴行业'
# 删除异常数据
all_data = all_data[-(all_data['经营单位'].str.contains('REOO-上海润诚|RFOO-北京京信'))]


#
# 重置index
all_data.reset_index(drop = True,inplace = True)

# 新建一个列表存储浙江省地级市
city = ['杭州市','宁波市','温州市','嘉兴市','湖州市','绍兴市','金华市','衢州市','舟山市','台州市','丽水市']
# 新增一列城市字段
all_data['城市'] = None

# 根据经营单位和浙江省地级市列表为城市赋值
for i in range(len(all_data)):
    tmp = all_data.loc[i,'经营单位']
    for j in range(len(city)):
        if city[j][:2] in tmp:
            all_data.loc[i,'城市'] = city[j]

print(type(all_data.loc[i,'经营单位']))

# 根据经营单位进行数据筛选，然后城市字段进行赋值
all_data.loc[all_data['经营单位'].str.contains('新昌新苗'),'城市'] = '绍兴市'
all_data.loc[all_data['经营单位'].str.contains('义乌军梦供应链'),'城市'] = '金华市'
all_data.loc[all_data['经营单位'].str.contains('浙江京城网络科技|浙江百诚|百诚网络|百诚音响|德诚网络科技|浙江五星电器|国大商贸|浙江千诚|索嘉贸易|浙江卓诚数码|信诚电器|百诚超市|百诚售后|世纪百诚|同和塑业|百诚未莱|怡亚通浙江杭州分公司|杭州佳宝|杭州万鸿供应链|杭州兴禾供应链'),'城市']='杭州市'
all_data.loc[all_data['经营单位'].str.contains('浙江国商|台州国兴|台州密森'),'城市'] = '台州市'
all_data.loc[all_data['经营单位'].str.contains('温州百诚|温州怡亚通|温州瑞家供应链|温州嘉源'),'城市'] = '温州市'
all_data.loc[all_data['经营单位'].str.contains('嘉兴百诚|浙江永润'),'城市'] = '嘉兴市'
all_data.loc[all_data['经营单位'].str.contains('浙江百城物流'),'城市'] = '潮州市'

all_data['采购量'] = None
all_data['采购量'] = all_data['采购额'] / all_data['商品指导采购价']

all_data['库存周转量'] = None
all_data['库存周转量'] = all_data['采购量'] / all_data['商品毛重']

import numpy as np
all_data.loc[(all_data['库存周转量'].isnull()),'库存周转量'] = 0.0
all_data.loc[all_data['库存周转量'] == np.inf,'库存周转量'] = 0.0

# 保存数据
all_data.to_csv('D:\\@我的记录文件夹\\临时文件夹\\clean_data.csv',index = False)

# 数据准备
gb_pur = all_data.groupby('行业').agg({'采购额':sum})
t = [(a,int(b)) for a,b in zip(gb_pur.index,gb_pur.values)]

# 绘制图表
from pyecharts.charts import Pie
from pyecharts import options as opts

p = (
    Pie()
    .set_global_opts(title_opts = opts.TitleOpts(title = '浙江省各行业采购额占比饼图'),
                     legend_opts = opts.LegendOpts(pos_top = '50'))
    .add('',t,center = ['50%','50%'],radius = [50,80])
)
p.render() # 图表显示

# 数据准备
pt_c_i = pd.pivot_table(all_data,index = '城市',columns = '行业',values = '采购额',aggfunc = 'sum') # 统计各城市各行业采购额
pt_c_i = pt_c_i.T.cumsum().iloc[::-1] # 累加数据并颠倒顺序
pt_c_i = pt_c_i.apply(lambda x:x/100000) # 量纲的转换
pt_c_i

# 绘制图表
import matplotlib.pyplot as plt
colors = ['blue','orange','red','green','gray']
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize = (10,8))
plt.title('各城市各行业采购额占比柱状图')
plt.xlabel('城市')
plt.ylabel('采购额（十万元）')
for i in range(pt_c_i.shape[0]):
    plt.bar(pt_c_i.columns,pt_c_i.iloc[i,:],color = colors[i],width = 0.5)

plt.legend(pt_c_i.index,loc = 'upper left')
plt.show()

# 数据准备
city_pur = all_data.groupby('城市').agg({'采购额':sum})
city_pur = [(a,round(float(b)/100000,3)) for a,b in zip(city_pur.index,city_pur.values)] # 量纲和数据结构的抓换
# 绘制图表
from pyecharts.charts import Map
from pyecharts import options as opts

c = (
    Map()
    .add("",city_pur,maptype = '浙江')
    .set_global_opts(title_opts = opts.TitleOpts(title = '浙江省各城市采购额地图'),
                     legend_opts = opts.LegendOpts(is_show = False),
                     visualmap_opts = (opts.VisualMapOpts(max_ = 10000,min_ = 0)))
)
c.render()

# 数据准备
all_data['企业'] = all_data.平台.apply(lambda x:x.split('-')[1]) # 字符床分割
all_data.loc[all_data['企业'] == '杭州','企业'] = '怡亚通浙江分公司'
data_platform = all_data[-all_data['企业'] == '丽水阳光']
all_data['企业'].value_counts()
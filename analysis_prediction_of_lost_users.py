import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# 数据导入
data1 = pd.read_csv('D:\\@我的记录文件夹\\大学课程文件\\大三\\生产实习python深度学习\\运营商流失用户分析与预测\\data\\USER_INFO_M.csv',encoding = 'utf-8')

# 数据抽样
data_USER_ID = data1.loc[data1['MONTH_ID'] == 201603, ['USER_ID','IS_LOST']].drop_duplicates() # 获取用户ID列表
data_USER_ID = data_USER_ID.groupby('IS_LOST').sample(n=1000).reset_index() # 用户列表分层抽样
data = pd.DataFrame(columns = data1.columns) # 创建新列表
for ID in data_USER_ID['USER_ID']:
    temp = data1[data1['USER_ID'] == ID]
    data = pd.concat([data,temp]).reset_index(drop=True)


# 数据提取整合
data2 = pd.DataFrame(columns = data1.columns) # 创建新列表
for ID in data_USER_ID['USER_ID']:
    # 选择抽样之后的每个用户的三条数据中的一条第三个月的数据
    temp = data.loc[data['USER_ID'] == ID,:].reset_index()
    temp1 = temp.loc[temp['MONTH_ID'] == 201603,:]
    data2 = pd.concat([data2,temp1]).reset_index(drop=True)

    # 根据有3个月数据的用户更新抽样的用户数据，比如取平均值等等，保证数据具有完整性
    if temp.shape[0] == 3:
        # 求解合约数
        a = temp['IS_AGREE'][0]
        b = temp['IS_AGREE'][1]
        c = temp['IS_AGREE'][2]
        heyue_num = 0
        if a == 1 and b == 1 and c == 1:
            heyue_num = 1.5
        else:
            heyue_num = c - (a + b) / 2
        data2.loc[data2.shape[0] - 1, 'IS_AGREE'] = heyue_num

        # 合约到期时间
        if temp['AGREE_EXP_DATE'][2] != None:
            data2.loc[data2.shape[0] - 1, 'AGREE_EXP_DATE'] = temp['AGREE_EXP_DATE'][2] - 201603
        else:
            data2.loc[data2.shape[0] - 1, 'AGREE_EXP_DATE'] = -1

        # 信用等级
        a = temp['CREDIT_LEVEL'][0]
        b = temp['CREDIT_LEVEL'][1]
        c = temp['CREDIT_LEVEL'][2]
        data2.loc[data2.shape[0] - 1, 'CREDIT_LEVEL'] = (a + b + c) / 3

        # VIP等级
        if temp['VIP_LVL'][2] != None:
            a = temp['VIP_LVL'][0]
            b = temp['VIP_LVL'][1]
            c = temp['VIP_LVL'][2]
            VIP_lvl = 0
            if a == b and b == c:
                VIP_lvl = c
            else:
                VIP_lvl = c - (a + b) / 2
            data2.loc[data2.shape[0] - 1, 'VIP_LVL'] = VIP_lvl
        else:
            data2.loc[data2.shape[0] - 1, 'VIP_LVL'] = 0

        # 本月费用
        # mmean('ACCT_FEE',data2,temp)
        a = temp['ACCT_FEE'][0]
        b = temp['ACCT_FEE'][1]
        c = temp['ACCT_FEE'][2]
        data2.loc[data2.shape[0] - 1, 'ACCT_FEE'] = (a + b + c) / 3

        # 在网时长 INNET_MONTH
        # mmean('INNET_MONTH', data2, temp)
        a = temp['INNET_MONTH'][0]
        b = temp['INNET_MONTH'][1]
        c = temp['INNET_MONTH'][2]
        data2.loc[data2.shape[0] - 1, 'INNET_MONTH'] = (a + b + c) / 3

        # 本地通话时长 NO_ROAM_LOCAL_CALL_DURA  本地通话次数 NO_ROAM_LOCAL_CDR_NUM
        # mmean('NO_ROAM_LOCAL_CALL_DURA', data2, temp)
        a = temp['NO_ROAM_LOCAL_CALL_DURA'][0]
        b = temp['NO_ROAM_LOCAL_CALL_DURA'][1]
        c = temp['NO_ROAM_LOCAL_CALL_DURA'][2]
        data2.loc[data2.shape[0] - 1, 'NO_ROAM_LOCAL_CALL_DURA'] = (a + b + c) / 3

        # mmean('NO_ROAM_LOCAL_CDR_NUM', data2, temp)
        a = temp['NO_ROAM_LOCAL_CDR_NUM'][0]
        b = temp['NO_ROAM_LOCAL_CDR_NUM'][1]
        c = temp['NO_ROAM_LOCAL_CDR_NUM'][2]
        data2.loc[data2.shape[0] - 1, 'NO_ROAM_LOCAL_CDR_NUM'] = (a + b + c) / 3

        # 国内长途通话时长 NO_ROAM_GN_LONG_CALL_DURA   国内长途通话次数 NO_ROAM_GN_LONG_CDR_NUM
        # mmean('NO_ROAM_GN_LONG_CALL_DURA', data2, temp)
        a = temp['NO_ROAM_GN_LONG_CALL_DURA'][0]
        b = temp['NO_ROAM_GN_LONG_CALL_DURA'][1]
        c = temp['NO_ROAM_GN_LONG_CALL_DURA'][2]
        data2.loc[data2.shape[0] - 1, 'NO_ROAM_GN_LONG_CALL_DURA'] = (a + b + c) / 3

        # mmean('NO_ROAM_GN_LONG_CDR_NUM', data2, temp)
        a = temp['NO_ROAM_GN_LONG_CDR_NUM'][0]
        b = temp['NO_ROAM_GN_LONG_CDR_NUM'][1]
        c = temp['NO_ROAM_GN_LONG_CDR_NUM'][2]
        data2.loc[data2.shape[0] - 1, 'NO_ROAM_GN_LONG_CDR_NUM'] = (a + b + c) / 3

        # 国内漫游通话时长 GN_ROAM_CALL_DURA  国内漫游通话次数  GN_ROAM_CDR_NUM
        # mmean('GN_ROAM_CALL_DURA', data2, temp)
        a = temp['GN_ROAM_CALL_DURA'][0]
        b = temp['GN_ROAM_CALL_DURA'][1]
        c = temp['GN_ROAM_CALL_DURA'][2]
        data2.loc[data2.shape[0] - 1, 'GN_ROAM_CALL_DURA'] = (a + b + c) / 3

        # mmean('GN_ROAM_CDR_NUM', data2, temp)
        a = temp['GN_ROAM_CDR_NUM'][0]
        b = temp['GN_ROAM_CDR_NUM'][1]
        c = temp['GN_ROAM_CDR_NUM'][2]
        data2.loc[data2.shape[0] - 1, 'GN_ROAM_CDR_NUM'] = (a + b + c) / 3

        # 通话次数 CDR_NUM   通话时长 CALL_DURA
        # mmean('CDR_NUM', data2, temp)
        a = temp['CDR_NUM'][0]
        b = temp['CDR_NUM'][1]
        c = temp['CDR_NUM'][2]
        data2.loc[data2.shape[0] - 1, 'CDR_NUM'] = (a + b + c) / 3

        # mmean('CALL_DURA', data2, temp)
        a = temp['CALL_DURA'][0]
        b = temp['CALL_DURA'][1]
        c = temp['CALL_DURA'][2]
        data2.loc[data2.shape[0] - 1, 'CALL_DURA'] = (a + b + c) / 3

        # P2P_SMS_CNT_UP
        # mmean('P2P_SMS_CNT_UP', data2, temp)
        a = temp['P2P_SMS_CNT_UP'][0]
        b = temp['P2P_SMS_CNT_UP'][1]
        c = temp['P2P_SMS_CNT_UP'][2]
        data2.loc[data2.shape[0] - 1, 'P2P_SMS_CNT_UP'] = (a + b + c) / 3

        # TOTAL_FLUX
        # mmean('TOTAL_FLUX', data2, temp)
        a = temp['TOTAL_FLUX'][0]
        b = temp['TOTAL_FLUX'][1]
        c = temp['TOTAL_FLUX'][2]
        data2.loc[data2.shape[0] - 1, 'TOTAL_FLUX'] = (a + b + c) / 3

        # LOCAL_FLUX
        # mmean('LOCAL_FLUX', data2, temp)
        a = temp['LOCAL_FLUX'][0]
        b = temp['LOCAL_FLUX'][1]
        c = temp['LOCAL_FLUX'][2]
        data2.loc[data2.shape[0] - 1, 'LOCAL_FLUX'] = (a + b + c) / 3

        # GN_ROAM_FLUX
        # mmean('GN_ROAM_FLUX', data2, temp)
        a = temp['GN_ROAM_FLUX'][0]
        b = temp['GN_ROAM_FLUX'][1]
        c = temp['GN_ROAM_FLUX'][2]
        data2.loc[data2.shape[0] - 1, 'GN_ROAM_FLUX'] = (a + b + c) / 3

        # CALL_DAYS
        # mmean('CALL_DAYS', data2, temp)
        a = temp['CALL_DAYS'][0]
        b = temp['CALL_DAYS'][1]
        c = temp['CALL_DAYS'][2]
        data2.loc[data2.shape[0] - 1, 'CALL_DAYS'] = (a + b + c) / 3

        # CALLING_DAYS
        # mmean('CALLING_DAYS', data2, temp)
        a = temp['CALLING_DAYS'][0]
        b = temp['CALLING_DAYS'][1]
        c = temp['CALLING_DAYS'][2]
        data2.loc[data2.shape[0] - 1, 'CALLING_DAYS'] = (a + b + c) / 3

        # CALLED_DAYS
        # mmean('CALLED_DAYS', data2, temp)
        a = temp['CALLED_DAYS'][0]
        b = temp['CALLED_DAYS'][1]
        c = temp['CALLED_DAYS'][2]
        data2.loc[data2.shape[0] - 1, 'CALLED_DAYS'] = (a + b + c) / 3
        # CALL_RING
        # mmean('CALL_RING', data2, temp)
        a = temp['CALL_RING'][0]
        b = temp['CALL_RING'][1]
        c = temp['CALL_RING'][2]
        data2.loc[data2.shape[0] - 1, 'CALL_RING'] = (a + b + c) / 3
        # CALLING_RING
        # mmean('CALLING_RING', data2, temp)
        a = temp['CALLING_RING'][0]
        b = temp['CALLING_RING'][1]
        c = temp['CALLING_RING'][2]
        data2.loc[data2.shape[0] - 1, 'CALLING_RING'] = (a + b + c) / 3
        # CALLED_RING
        # mmean('CALLED_RING', data2, temp)
        a = temp['CALLED_RING'][0]
        b = temp['CALLED_RING'][1]
        c = temp['CALLED_RING'][2]
        data2.loc[data2.shape[0] - 1, 'CALLED_RING'] = (a + b + c) / 3
data2 = data2[(data2['INNET_MONTH'] >= 0) & (data2['INNET_MONTH'] <= 40000)]

# 数据清洗
# 重复数据删除
data2 = data2.drop_duplicates()

# 缺失值处理
data2['CUST_SEX'].fillna(value = 3,inplace = True)
data2['CERT_AGE'].fillna(value = 0,inplace = True)
data2['CONSTELLATION_DESC'].fillna(value = 0,inplace = True)
data2['AGREE_EXP_DATE'].fillna(value = 0,inplace = True)
data2['VIP_LVL'].fillna(value = 0,inplace = True)
data2['OS_DESC'].fillna(value = 'unknown',inplace = True)



# 第一部分的特征选择(先除去'手机型号名称','操作系统','星座','品牌','终端'进行第一部分特征选择)
# data3用于第一部分数据的特征选择
data3 = data2.drop('USER_ID',axis = 1) # 删除用户ID
data3 = data3.drop('MODEL_NAME',axis = 1) # 删除手机型号名称
data3 = data3.drop('OS_DESC',axis = 1) # 删除操作系统
data3 = data3.drop('CONSTELLATION_DESC',axis = 1) # 删除星座
data3 = data3.drop('MANU_NAME',axis = 1) # 删除品牌
data3 = data3.drop('TERM_TYPE',axis = 1) # 删除终端

# 数据集分割
# 获取所有列的列表
all_columns = data3.columns.tolist()
# 选择大部分列（排除IS_LOST列）
selected_columns = all_columns[:-2]
labels = data3.loc[:,'IS_LOST']
data_train, data_test, labels_train, labels_test = train_test_split(data2.loc[:,selected_columns],labels,test_size=0.20, random_state=100)

# 创建随机森林分类器对象
rf = RandomForestClassifier()
# 使用训练数据拟合模型
rf.fit(data_train, labels_train)
# 获取特征重要性得分
importance_scores = rf.feature_importances_
# 构建特征名和重要程度的元组列表
feature_importances = [(feature, importance) for feature, importance in zip(data3.columns, importance_scores)]
# 按特征的重要程度进行降序排序
feature_importances.sort(key=lambda x: x[1], reverse=True)
# 显示特征的重要程度
for feature, importance in feature_importances:
    print(f"{feature}: {importance}")

# 画出重要程度累计图
sum_lst = list(range(0,len(feature_importances) + 1))
for i in range(0,len(feature_importances)):
    sum_lst[i + 1] = feature_importances[i][1] + sum_lst[i]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体
plt.plot(list(range(1,len(feature_importances) + 1)),sum_lst[0:len(sum_lst) - 1])
plt.xlabel('选择特征个数')
plt.ylabel('特征重要程度累计和')
plt.title("特征重要程度累计图")
plt.show()


# 选择前20个特征
mlen = len(feature_importances)
for i in range(0,mlen - 20):
    data2 = data2.drop(feature_importances[mlen - i - 1][0], axis=1)





# 星座直方图
grouped_1 = data2.groupby('CONSTELLATION_DESC')['IS_LOST'].apply(lambda x: x[x == 0].value_counts()).to_frame()
grouped_1 = grouped_1['IS_LOST'].div(grouped_1['IS_LOST'].sum())
grouped_2 = data2.groupby('CONSTELLATION_DESC')['IS_LOST'].apply(lambda x: x[x == 1].value_counts()).to_frame()
grouped_2 = grouped_2['IS_LOST'].div(grouped_2['IS_LOST'].sum())

grouped = pd.merge(grouped_1, grouped_2, left_on='CONSTELLATION_DESC', right_on='CONSTELLATION_DESC',how='outer')
grouped = grouped.fillna(value={'IS_LOST_x': 0, 'IS_LOST_y': 0})
grouped['CONSTELLATION_DESC'] = grouped.index

# 获取分类和数值列名
categories = grouped['CONSTELLATION_DESC']
values = grouped.drop('CONSTELLATION_DESC', axis=1)

# 设置柱状图的宽度
bar_width = 0.2

# 计算每个柱状图的位置
r1 = range(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体

# 绘制柱状图
plt.bar(r1, values['IS_LOST_x'], color='b', width=bar_width, label='非流失用户')
plt.bar(r2, values['IS_LOST_y'], color='g', width=bar_width, label='流失用户')

# 添加刻度标签和图例
plt.xticks([r + bar_width for r in range(len(categories))], categories)
plt.xlabel('星座')
plt.ylabel('所占流失/非流失用户总人数比例')
plt.title('流失与非流失用户星座频率直方图')
plt.legend()

# 显示图形
plt.show()



# 手机品牌名称直方图
grouped_1 = data2.groupby('MANU_NAME')['IS_LOST'].apply(lambda x: x[x == 0].value_counts()).to_frame()
grouped_1 = grouped_1['IS_LOST'].div(grouped_1['IS_LOST'].sum())
grouped_2 = data2.groupby('MANU_NAME')['IS_LOST'].apply(lambda x: x[x == 1].value_counts()).to_frame()
grouped_2 = grouped_2['IS_LOST'].div(grouped_2['IS_LOST'].sum())

grouped = pd.merge(grouped_1, grouped_2, left_on='MANU_NAME', right_on='MANU_NAME',how='outer')
grouped = grouped.fillna(value={'IS_LOST_x': 0, 'IS_LOST_y': 0})
grouped['MANU_NAME'] = grouped.index

# 获取分类和数值列名
categories = grouped['MANU_NAME']
values = grouped.drop('MANU_NAME', axis=1)

# 设置柱状图的宽度
bar_width = 0.4

# 计算每个柱状图的位置
r1 = range(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体

# 绘制柱状图
plt.bar(r1, values['IS_LOST_x'], color='b', width=bar_width, label='非流失用户')
plt.bar(r2, values['IS_LOST_y'], color='g', width=bar_width, label='流失用户')

# 添加刻度标签和图例
plt.xticks([r + bar_width for r in range(len(categories))], categories)
plt.xlabel('手机品牌名称')
plt.ylabel('所占流失/非流失用户总人数比例')
plt.title('流失与非流失用户手机品牌频率直方图')
plt.legend()

# 显示图形
plt.show()

# 型号直方图
grouped_1 = data2.groupby('MODEL_NAME')['IS_LOST'].apply(lambda x: x[x == 0].value_counts()).to_frame()
grouped_1 = grouped_1['IS_LOST'].div(grouped_1['IS_LOST'].sum())
grouped_2 = data2.groupby('MODEL_NAME')['IS_LOST'].apply(lambda x: x[x == 1].value_counts()).to_frame()
grouped_2 = grouped_2['IS_LOST'].div(grouped_2['IS_LOST'].sum())

grouped = pd.merge(grouped_1, grouped_2, left_on='MODEL_NAME', right_on='MODEL_NAME',how='outer')
grouped = grouped.fillna(value={'IS_LOST_x': 0, 'IS_LOST_y': 0})
grouped['MODEL_NAME'] = grouped.index

# 获取分类和数值列名
categories = grouped['MODEL_NAME']
values = grouped.drop('MODEL_NAME', axis=1)

# 设置柱状图的宽度
bar_width = 0.2

# 计算每个柱状图的位置
r1 = range(len(categories))
r2 = [x + bar_width for x in r1]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体

# 绘制柱状图
plt.bar(r1, values['IS_LOST_x'], color='b', width=bar_width, label='非流失用户')
plt.bar(r2, values['IS_LOST_y'], color='g', width=bar_width, label='流失用户')

# 添加刻度标签和图例
plt.xticks([r + bar_width for r in range(len(categories))], categories)
plt.xlabel('手机型号')
plt.ylabel('所占流失/非流失用户总人数比例')
plt.title('流失与非流失用户手机型号频率直方图')
plt.legend()

# 显示图形
plt.show()



# 操作系统直方图
grouped_1 = data2.groupby('OS_DESC')['IS_LOST'].apply(lambda x: x[x == 0].value_counts()).to_frame()
grouped_1 = grouped_1['IS_LOST'].div(grouped_1['IS_LOST'].sum())
grouped_2 = data2.groupby('OS_DESC')['IS_LOST'].apply(lambda x: x[x == 1].value_counts()).to_frame()
grouped_2 = grouped_2['IS_LOST'].div(grouped_2['IS_LOST'].sum())

grouped = pd.merge(grouped_1, grouped_2, left_on='OS_DESC', right_on='OS_DESC',how='outer')
grouped = grouped.fillna(value={'IS_LOST_x': 0, 'IS_LOST_y': 0})
grouped['OS_DESC'] = grouped.index

# 获取分类和数值列名
categories = grouped['OS_DESC']
values = grouped.drop('OS_DESC', axis=1)

# 设置柱状图的宽度
bar_width = 0.4

# 计算每个柱状图的位置
r1 = range(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体

# 绘制柱状图
plt.bar(r1, values['IS_LOST_x'], color='b', width=bar_width, label='非流失用户')
plt.bar(r2, values['IS_LOST_y'], color='g', width=bar_width, label='流失用户')

# 添加刻度标签和图例
plt.xticks([r + bar_width for r in range(len(categories))], categories)
plt.xlabel('OS')
plt.ylabel('所占流失/非流失用户总人数比例')
plt.title('流失与非流失用户OS频率直方图')
plt.legend()

# 显示图形
plt.show()


# 终端硬件类型直方图
grouped_1 = data2.groupby('TERM_TYPE')['IS_LOST'].apply(lambda x: x[x == 0].value_counts()).to_frame()
grouped_1 = grouped_1['IS_LOST'].div(grouped_1['IS_LOST'].sum())
grouped_2 = data2.groupby('TERM_TYPE')['IS_LOST'].apply(lambda x: x[x == 1].value_counts()).to_frame()
grouped_2 = grouped_2['IS_LOST'].div(grouped_2['IS_LOST'].sum())

grouped = pd.merge(grouped_1, grouped_2, left_on='TERM_TYPE', right_on='TERM_TYPE',how='outer')
grouped = grouped.fillna(value={'IS_LOST_x': 0, 'IS_LOST_y': 0})
grouped['TERM_TYPE'] = grouped.index

# 获取分类和数值列名
categories = grouped['TERM_TYPE']
values = grouped.drop('TERM_TYPE', axis=1)

# 设置柱状图的宽度
bar_width = 0.4

# 计算每个柱状图的位置
r1 = range(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体

# 绘制柱状图
plt.bar(r1, values['IS_LOST_x'], color='b', width=bar_width, label='非流失用户')
plt.bar(r2, values['IS_LOST_y'], color='g', width=bar_width, label='流失用户')

# 添加刻度标签和图例
plt.xticks([r + bar_width for r in range(len(categories))], categories)
plt.xlabel('终端硬件类型')
plt.ylabel('所占流失/非流失用户总人数比例')
plt.title('流失与非流失用户终端硬件类型频率直方图')
plt.legend()

# 显示图形
plt.show()



# 根据主观和上述直方图进行另一部分的特征选择
if 'MONTH_id' in data2.columns:
    data2 = data2.drop('MONTH_ID',axis = 1) # 删除月份
if 'USER_ID' in data2.columns:
    data2 = data2.drop('USER_ID',axis = 1) # 删除用户ID
if 'CONSTELLATION_DESC' in data2.columns:
    data2 = data2.drop('CONSTELLATION_DESC',axis = 1) # 删除星座
if 'MODEL_NAME' in data2.columns:
    data2 = data2.drop('MODEL_NAME',axis = 1) # 删除手机型号名称
if 'MANU_NAME' in data2.columns:
    data2 = data2.drop('MANU_NAME',axis = 1) # 删除品牌

# 对OS操作系统和终端硬件进行哑变量处理
dummy_df = pd.get_dummies(data2['OS_DESC'])
data2 = data2.drop('OS_DESC',axis = 1)
data2 = pd.concat([data2, dummy_df], axis=1)

dummy_df = pd.get_dummies(data2['TERM_TYPE'])
data2 = data2.drop('TERM_TYPE',axis = 1)
data2 = pd.concat([data2, dummy_df], axis=1)


data2.columns = data2.columns.astype(str)

# 数据集分割
# 获取所有列的列表
all_columns = data2.columns.tolist()
# 选择大部分列（排除IS_LOST列）
selected_columns = all_columns[:-2]
labels = data2.loc[:,'IS_LOST']
data_train, data_test, labels_train, labels_test = train_test_split(data2.loc[:,selected_columns],labels,test_size=0.20, random_state=100)
# 使用标准化(Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(data_train)
X_test_scaled = scaler.transform(data_test)

# 模型训练
# 支持向量机模型
model_1 = svm.SVC(probability=True)
model_1.fit(data_train,labels_train)
labels_pre_1 = model_1.predict(data_test)
print(labels_pre_1,'\n',labels_test.values)
# 模型评价
accuracy_1 = accuracy_score(labels_test, labels_pre_1)
precision_1 = precision_score(labels_test, labels_pre_1)
recall_1 = recall_score(labels_test, labels_pre_1)
f1_1 = f1_score(labels_test, labels_pre_1)
labels_scores_1 = model_1.predict_proba(data_test)[:, 1]
auc_1 = roc_auc_score(labels_test, labels_scores_1)

# 朴素贝叶斯模型
model_2 = GaussianNB()
model_2.fit(data_train,labels_train)
labels_pre_2 = model_2.predict(data_test)
print(labels_pre_2,'\n',labels_test.values)
# 模型评价
accuracy_2 = accuracy_score(labels_test, labels_pre_2)
precision_2 = precision_score(labels_test, labels_pre_2)
recall_2 = recall_score(labels_test, labels_pre_2)
f1_2 = f1_score(labels_test, labels_pre_2)
labels_scores_2 = model_2.predict_proba(data_test)[:, 1]
auc_2 = roc_auc_score(labels_test, labels_scores_2)

# 决策树模型
model_3 = DecisionTreeClassifier()
model_3.fit(data_train,labels_train)
labels_pre_3 = model_3.predict(data_test)
print(labels_pre_3,'\n',labels_test.values)
# 模型评价
accuracy_3 = accuracy_score(labels_test, labels_pre_3)
precision_3 = precision_score(labels_test, labels_pre_3)
recall_3 = recall_score(labels_test, labels_pre_3)
f1_3 = f1_score(labels_test, labels_pre_3)
labels_scores_3 = model_3.predict_proba(data_test)[:, 1]
auc_3 = roc_auc_score(labels_test, labels_scores_3)

# 画模型评价折线图
x = ['准确率','精确率','召回率','F1','AUC值']
y1 = [accuracy_1,precision_1,recall_1,f1_1,auc_1]
y2 = [accuracy_2,precision_2,recall_2,f1_2,auc_2]
y3 = [accuracy_3,precision_3,recall_3,f1_3,auc_3]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为宋体或其他支持中文的字体

plt.plot(x, y1, label='支持向量机模型')
plt.plot(x, y2, label='朴素贝叶斯模型')
plt.plot(x, y3, label='决策树模型')

plt.legend()

plt.title('模型评价对比折线图')
plt.xlabel('模型评价标准')
plt.ylabel('评价值')
plt.show()

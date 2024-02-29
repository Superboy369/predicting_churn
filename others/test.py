import pandas as pd

# 创建包含数字字符串的 DataFrame
data = {'A': ['1', '2', '3', '4', '5'],
        'B': ['6.7', '8.9', '10.2', '12.3', '14.5']}
df = pd.DataFrame(data)

# 将列'A'中的数字字符串转换为整数
df['A'] = df['A'].astype(int)

# 将列'B'中的数字字符串转换为浮点数
df['B'] = df['B'].astype(float)

print(df)
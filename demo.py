import re
import pandas as pd


def pattern_table(table_string):
    # 用正则表达式解析表格
    pattern = re.compile(r'\| (\w+) \| +([\d.]+) +\| +([\d.]+) +\| +([\d.]+) +\| +([\d.]+) +\|')
    matches = pattern.findall(table_string)

    # 将匹配结果转为DataFrame
    data = [[match[0], float(match[1]), float(match[2]), float(match[3]), float(match[4])] for match in matches]
    df = pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'Specificity', 'F1 Score'])
    return df


table_string1 = """
+-----+-----------+--------+-------------+----------+
|     | Precision | Recall | Specificity | F1 Score |
+-----+-----------+--------+-------------+----------+
| Inc |   0.685   | 0.756  |    0.913    |  0.719   |
| ISC |   0.724   |  0.62  |    0.941    |  0.668   |
| Noi |   0.967   | 0.966  |    0.992    |  0.966   |
| Nor |   0.936   | 0.968  |    0.984    |  0.952   |
| Sti |   0.953   | 0.961  |    0.988    |  0.957   |
+-----+-----------+--------+-------------+----------+
"""
table_string2 = """
+-----+-----------+--------+-------------+----------+
|     | Precision | Recall | Specificity | F1 Score |
+-----+-----------+--------+-------------+----------+
| Inc |   0.691   | 0.851  |    0.905    |  0.763   |
| ISC |   0.801   | 0.595  |    0.963    |  0.683   |
| Noi |   0.846   | 0.999  |    0.955    |  0.916   |
| Nor |   0.992   | 0.823  |    0.998    |   0.9    |
| Sti |   0.981   | 0.995  |    0.995    |  0.988   |
+-----+-----------+--------+-------------+----------+
"""
table_string3 = """
+-----+-----------+--------+-------------+----------+
|     | Precision | Recall | Specificity | F1 Score |
+-----+-----------+--------+-------------+----------+
| Inc |   0.635   | 0.668  |    0.904    |  0.651   |
| ISC |    0.65   | 0.609  |    0.918    |  0.629   |
| Noi |   0.994   | 0.993  |    0.999    |  0.993   |
| Nor |   0.992   | 0.994  |    0.998    |  0.993   |
| Sti |   0.988   | 0.999  |    0.997    |  0.993   |
+-----+-----------+--------+-------------+----------+
"""
table1 = pattern_table(table_string1)
table2 = pattern_table(table_string2)
table3 = pattern_table(table_string3)

# 合并三个表格的数据
all_tables = [table1, table2, table3]
merged_df = pd.concat(all_tables)

# 按类别计算平均值
avg_values = merged_df.groupby('Class').mean()

# 打印平均值
print("Average Values:")
print(avg_values)


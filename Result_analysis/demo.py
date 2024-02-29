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
| Inc |   0.907   | 0.808  |    0.979    |  0.855   |
| ISC |   0.878   | 0.841  |    0.971    |  0.859   |
| Noi |   0.977   |  1.0   |    0.994    |  0.988   |
| Nor |   0.851   | 0.975  |    0.957    |  0.909   |
| Sti |   0.842   | 0.828  |    0.961    |  0.835   |
+-----+-----------+--------+-------------+----------+
"""
table_string2 = """
+-----+-----------+--------+-------------+----------+
|     | Precision | Recall | Specificity | F1 Score |
+-----+-----------+--------+-------------+----------+
| Inc |   0.935   | 0.925  |    0.984    |   0.93   |
| ISC |   0.992   | 0.762  |    0.998    |  0.862   |
| Noi |   0.918   |  1.0   |    0.978    |  0.957   |
| Nor |   0.849   | 0.926  |    0.959    |  0.886   |
| Sti |   0.781   |  0.83  |    0.942    |  0.805   |
+-----+-----------+--------+-------------+----------+
"""
table_string3 = """
+-----+-----------+--------+-------------+----------+
|     | Precision | Recall | Specificity | F1 Score |
+-----+-----------+--------+-------------+----------+
| Inc |   0.898   | 0.814  |    0.977    |  0.854   |
| ISC |   0.845   | 0.823  |    0.962    |  0.834   |
| Noi |   0.943   |  1.0   |    0.985    |  0.971   |
| Nor |   0.841   | 0.932  |    0.956    |  0.884   |
| Sti |   0.866   | 0.823  |    0.968    |  0.844   |
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


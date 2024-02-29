import math

def calculate_standard_deviation(accuracies):
    # 计算平均值
    mean_accuracy = sum(accuracies) / len(accuracies)

    # 计算方差
    variance = sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)

    # 计算标准差
    standard_deviation = math.sqrt(variance)

    return standard_deviation

def main():
    accuracies = [0.8458, 0.7917, 0.8282]

    # 计算标准差
    std_dev = calculate_standard_deviation(accuracies)

    # 输出结果
    print("准确率: {:.2f} ± {:.2f}".format(sum(accuracies) / len(accuracies) * 100, std_dev * 100))

if __name__ == "__main__":
    main()

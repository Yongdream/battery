import pickle


def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("文件不存在：", file_path)
        return None
    except Exception as e:
        print("读取文件时出错：", e)
        return None


if __name__ == "__main__":
    file_path = r"E:/Galaxy/data/battery_brand3/data/0.pkl"
    data = read_pickle_file(file_path)
    if data is not None:
        print("读取成功！数据内容如下：")
        print(data)

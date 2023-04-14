import torch

from my_dataset import MyDataSet
import time
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = r'..\processed\udds'
batch_size = 16


def validate_model(model: torch.nn.Module):
    validate_dataset = MyDataSet(root + "val")
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        t1 = time.time()
        for val_data in tqdm(validate_loader, desc="validate model accuracy."):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.sum(torch.eq(predict_y, val_labels.to(device))).item()
        val_accurate = acc / val_num
        print('test_accuracy: %.3f, time:%.3f' % (val_accurate, time.time() - t1))

    return val_accurate


def main():
    print('pending')

if __name__ == '__main__':
    main()
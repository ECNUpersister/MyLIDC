from torch.utils.data import DataLoader
from engine import *


def main():
    device = torch.device('cuda')
    dataset_name = 'G:/MyLIDC/data/dataset/lidc_shape64'
    dataset_train, dataset_test = get_dataset(dataset_name, sample_ratio=0.1)

    data_loader_train = DataLoader(dataset_train,batch_size=3,shuffle=True,collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test,batch_size=3,shuffle=False,collate_fn=collate_fn)

    model = get_maskrcnn(pretrained=False).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.05, weight_decay=0.0005) # adam有时会出现nan导致不能优化
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(10):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device, shape=64)

    print("运行完成！")


if __name__ == '__main__':
    main()

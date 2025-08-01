import cv2

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

test_data = dataset.MNIST(
    root = "mnist",
    train = False,
    transform = transforms.ToTensor(),
    download = True
);

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)




# 修正模型加载：添加 weights_only=False
try:
    cnn = torch.load("model/mnist_model.pkl", weights_only=False, map_location=torch.device('cpu'))
except Exception as e:
    print(f"模型加载失败：{e}")
    raise





# # 加载已经训练好的模型文件
# cnn = torch.load("model/mnist_model.pkl")
loss_test = 0
rightValue = 0
loss_func = torch.nn.CrossEntropyLoss()
for index, (images, labels) in enumerate(test_loader):
    ##前向传播
    outputs = cnn(images)  ##数据扔进神经函数中
    _,pred = outputs.max(1)
    loss_test += loss_func(outputs, labels)
    rightValue += (pred == labels).sum().item()

    #改变格式
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    #遍历每一个样本
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]

        print("预测值为{}".format(im_pred))
        print("真实值为{}".format(im_label))



        # 调整图像维度和类型以适应OpenCV
        # 1. 去除通道维度: [1, 28, 28] -> [28, 28]
        # 2. 转换为uint8类型 (0-255范围)
        display_image = im_data[0] * 255  # 提取单通道并缩放至0-255
        display_image = display_image.astype('uint8')  # 转换为uint8类型

        cv2.namedWindow("nowImage", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("nowImage", 400, 400)

        # 显示图像
        cv2.imshow("nowImage", display_image)
        cv2.waitKey(0)




    print("loss为{}，准确率是{}".format(loss_test, rightValue / len(test_data)))





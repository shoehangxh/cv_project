import main
import torch

checkpoint = torch.load('./checkpoint/cifar10_epoch_5.ckpt')
main.net.load_state_dict(checkpoint['net'])

correct = 0
total = 0
with torch.no_grad():
    for data in main.testloader:
        images, labels = data
        outputs = main.net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in main.testloader:
        images, labels = data
        outputs = main.net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s: %2d %%' %
          (cifar10_classes[i], 100* class_correct[i] / class_total[i]))



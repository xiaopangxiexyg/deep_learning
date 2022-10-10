### darknet.py
- 定义一个类，初始化神经网络
  1. Conv2d:
   torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
   参数：输入通道数、输出通道数、卷积核尺寸( n:n*n;or(a,b):a*b )、卷积步长、填充操作、padding模式、扩张操作、控制分组卷积、为真，则在输出中添加一个可学习的偏差
  2. cudnn报错：减小batchsize 以及运行代码前先杀掉僵尸进程
  3. 
#### pytprch基础语法
pytorch的最基本的操作是Tensor<br>
不同数据类型的Tensor：torch.FloatTensor torch.DoubleTensor torch.ShortTensor torch.IntTensor toch.longTensor
* 创建一个三行两列给定元素的矩阵
```
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is:{}'.format(a))

out:
a is:tensor([[2., 3.],
        [4., 8.],
        [7., 9.]])
```

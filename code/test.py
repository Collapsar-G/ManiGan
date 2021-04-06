import numpy as np
import torch
from torch.autograd import Variable

# masks = []
# batch_size = 5
# class_ids = Variable(torch.LongTensor(range(batch_size)))
# print(class_ids)
# for i in range(batch_size):
#     mask = (class_ids == class_ids[i])
#     print(mask)
#     mask[i] = 0
#     print(mask,"#########33")
#     masks.append(mask.reshape((1, -1)))
# print(masks,"!!!!!!!!!!!!11")
# masks = np.concatenate(masks, 0)
# masks = torch.ByteTensor(masks)
# print(masks)
context = torch.from_numpy(np.zeros((4,3,5,6)))
print(context.size(2))
print(context.view(4, -1, 30).size(2))

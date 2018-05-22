import torch
from torch import FloatTensor
from torch.autograd import Variable
from timeit import default_timer as timer
from torch.nn.functional import grid_sample
from torch.nn import ReplicationPad2d


def grid_bilinear_sampling(A, x, y):
    batch_size, k, h, w = A.size()
    x_norm = x/((w-1)/2) - 1
    y_norm = y/((h-1)/2) - 1
    grid = torch.cat((x_norm.view(batch_size, h, w, 1), y_norm.view(batch_size, h, w, 1)), 3)
    Q = grid_sample(A, grid, mode='bilinear')
    in_view_mask = Variable(((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data))
    # in_view_mask = Variable(((x.data > 1) & (x.data < w-2) & (y.data > 1) & (y.data < h-2)).type_as(A.data))
    # in_view_mask = Variable(((x.data > -1+3/w) & (x.data < 1-3/w) & (y.data > -1+3/h) & (y.data < 1-3/h)).type_as(A.data))
    return Q.view(batch_size, k, h*w), in_view_mask


#
# def bilinear_sampling_batch(A, x, y):
#     batch_size, k, h, w = A.size()
#     _, N = x.size()
#
#     # s = timer()
#     in_view_mask = Variable(((x.data >= 0) & (x.data <= w-1) & (y.data >= 0) & (y.data <= h-1)).type_as(A.data))
#     x = x.clamp(0, w-1)
#     y = y.clamp(0, h-1)
#
#     x1 = x.floor().clamp(0, w-2)
#     x2 = x1+1
#     y1 = y.floor().clamp(0, h-2)
#     y2 = y1+1
#
#     x_minus_x1 = (x-x1).view(batch_size, 1, N)
#     x2_minus_x = (x2-x).view(batch_size, 1, N)
#
#     y_minus_y1 = (y-y1).view(batch_size, 1, N)
#     y2_minus_y = (y2-y).view(batch_size, 1, N)
#
#     # print(timer()-s)
#
#     # s = timer()
#     A = A.view(batch_size, k, h*w).permute(1, 0, 2).contiguous().view(k, h*w*batch_size)
#
#
#     index_batch_term = torch.range(0,(batch_size-1)*N, N).view(batch_size, 1).expand(batch_size, N)
#     y2_times_w = y2.data*w
#     y1_times_w = y1.data*w
#
#     # print(timer() - s)
#
#     # s = timer()
#     index = (x1.data+y1_times_w) + index_batch_term
#     index = Variable(index.long().view(batch_size*N))
#     # print(timer()-s)
#     Q11 = A.index_select(1, index).view(k, batch_size, N).permute(1, 0, 2)
#     # print(timer()-s)
#
#     index = (x1.data+y2_times_w) + index_batch_term
#     index = Variable(index.long().view(batch_size*N))
#     Q12 = A.index_select(1, index).view(k, batch_size, N).permute(1, 0, 2)
#
#     index = (x2.data+y1_times_w) + index_batch_term
#     index = Variable(index.long().view(batch_size*N))
#     Q21 = A.index_select(1, index).view(k, batch_size, N).permute(1, 0, 2)
#
#     index = (x2.data+y2_times_w) + index_batch_term
#     index = Variable(index.long().view(batch_size*N))
#     Q22 = A.index_select(1, index).view(k, batch_size, N).permute(1, 0, 2)
#
#     # print(timer()-s)
#
#     # s = timer()
#     Q = Q11*(x2_minus_x * y2_minus_y).expand_as(Q11) \
#         + Q21 * (x_minus_x1 * y2_minus_y).expand_as(Q21) \
#         + Q12 * (x2_minus_x * y_minus_y1).expand_as(Q12) \
#         + Q22 * (x_minus_x1 * y_minus_y1).expand_as(Q22)
#
#     # print(timer()-s)
#
#     # Q = Q * in_view_mask.view(batch_size, 1, N).expand_as(Q)
#     return Q, in_view_mask
#
#
#
# def bilinear_sampling(A, x, y):
#     k, h, w = A.size()
#     N = x.numel()
#
#     in_view_mask = Variable(((x.data > 0)*(x.data < w-1)*(y.data > 0)*(y.data < h-1)).type_as(A.data))
#     x = x.clamp(0, w-1)
#     y = y.clamp(0, h-1)
#
#     x1 = x.floor().clamp(0, w-2)
#     x2 = x1+1
#     y1 = y.floor().clamp(0, h-2)
#     y2 = y1+1
#
#     x_minus_x1 = (x-x1).view(1, N)
#     x2_minus_x = (x2-x).view(1, N)
#
#     y_minus_y1 = (x-x1).view(1, N)
#     y2_minus_y = (y2-y).view(1, N)
#
#     A = A.view(k, h*w)
#     index = Variable(x1.data+y1.data*w).long()
#     Q11 = A.index_select(1, index)
#
#     index = Variable(x1.data+y2.data*w).long()
#     Q12 = A.index_select(1, index)
#
#     index = Variable(x2.data+y1.data*w).long()
#     Q21 = A.index_select(1, index)
#
#     index = Variable(x2.data+y2.data*w).long()
#     Q22 = A.index_select(1, index)
#
#     Q = Q11*(x2_minus_x * y2_minus_y).expand_as(Q11) \
#         + Q21 * (x_minus_x1 * y2_minus_y).expand_as(Q21) \
#         + Q12 * (x2_minus_x * y_minus_y1).expand_as(Q12) \
#         + Q22 * (x_minus_x1 * y_minus_y1).expand_as(Q22)
#
#     Q = Q * in_view_mask.view(1, N).expand_as(Q)
#     return Q, in_view_mask


# test case
if __name__ == "__main__":
    # index = Variable(torch.randperm(4), requires_grad=True)
    # index_copy = Variable(index.data)
    # print(index)
    #
    # A = Variable(torch.randn(3,10), requires_grad=True)
    #
    # B = A.index_select(1, index_copy)
    #
    # print(B)
    #
    # B = B.clamp(0, 0.5)
    #
    # print(B)
    #
    # loss = B.mean()
    #
    #
    # loss.backward()
    # print(A.grad)




    # r = 120
    # batch_size = 5
    # x = Variable(torch.rand(batch_size, r*r)*r, requires_grad=True)
    # y = Variable(torch.rand(batch_size, r*r)*r, requires_grad=True)
    #
    # A = Variable(torch.rand(batch_size, 3,r,r), requires_grad=True)
    #
    # start = timer()
    # Q, M = grid_bilinear_sampling(A, x, y)
    #
    # loss = Q.mean()
    # loss.backward()
    #
    # end = timer()
    # print(end - start)


    tx = Variable(torch.ones(1), requires_grad=True)
    optimizer = torch.optim.SGD([tx], lr=.1)
    a = Variable(torch.sin(torch.arange(0, 3.14, .1)).unsqueeze(0).repeat(10, 1).unsqueeze(0).unsqueeze(0))

    h = 10
    w = 32

    x = Variable(torch.arange(0,32).unsqueeze(0).repeat(10,1))
    y = Variable(torch.arange(0,10).unsqueeze(1).repeat(1,32))

    print(x.size())
    print(y.size())

    tg_img, _ = grid_bilinear_sampling(a, x, y)
    tg_img = tg_img.view(h,w)
    for i in range(10):
        optimizer.zero_grad()
        src_img, M = grid_bilinear_sampling(a, x+tx, y)
        src_img = src_img.view(h, w)
        cost = ((src_img - tg_img)*M).norm()
        print(cost)
        print(src_img)
        print(tg_img)
        cost.backward()
        optimizer.step()
        print(M)
        print(tx)


    #
    # print(x.grad)
    # print(A.grad)

    # print(x)
    # print(y)
    # print(Q)

import torch
from torch import FloatTensor
from torch.autograd import Variable
from timeit import default_timer as timer
from torch.autograd import gradcheck

class Inverse(torch.autograd.Function):

    def forward(self, input):
        h, w = input.size()
        assert(h == w)
        H = input.inverse()
        self.save_for_backward(H)
        return H

    def backward(self, grad_output):
        # print(grad_output.is_contiguous())
        H, = self.saved_tensors
        h, w = H.size()
        assert(h == w)
        Hl = H.t().repeat(1, h).view(h*h, h, 1)
        # print(Hl.view(batch_size, h, h, h, 1))
        Hr = H.repeat(h, 1).view(h*h, 1, h)
        # print(Hr.view(batch_size, h, h, 1, h))

        r = Hl.bmm(Hr).view(h, h, h, h) * \
            grad_output.contiguous().view(1, 1, h, h).expand(h, h, h, h)
        # print(r.size())
        return -r.sum(-1).sum(-1)
        # print(r)

def inv(input):
    return Inverse()(input)

class InverseBatch(torch.autograd.Function):

    def forward(self, input):
        batch_size, h, w = input.size()
        assert(h == w)
        H = torch.Tensor(batch_size, h, h).type_as(input)
        for i in range(0, batch_size):
            H[i, :, :] = input[i, :, :].inverse()
        self.save_for_backward(H)
        return H

    def backward(self, grad_output):
        # print(grad_output.is_contiguous())
        H, = self.saved_tensors
        [batch_size, h, w] = H.size()
        assert(h == w)
        Hl = H.transpose(1,2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
        # print(Hl.view(batch_size, h, h, h, 1))
        Hr = H.repeat(1, h, 1).view(batch_size*h*h, 1, h)
        # print(Hr.view(batch_size, h, h, 1, h))

        r = Hl.bmm(Hr).view(batch_size, h, h, h, h) * \
            grad_output.contiguous().view(batch_size, 1, 1, h, h).expand(batch_size, h, h, h, h)
        # print(r.size())
        return -r.sum(-1).sum(-1)
        # print(r)

def inv_batch(input):
    return InverseBatch()(input)


if __name__ == "__main__":

    W = torch.rand(2,2)
    s = timer()
    invH = inv(Variable(W, requires_grad=True))
    print(timer() - s)
    c = invH.mean()
    print(c)
    c.backward()
    print(timer()-s)
    test = gradcheck(inv, (Variable(W, requires_grad=True),), eps=1e-5, atol=1e-4)
    print(test)

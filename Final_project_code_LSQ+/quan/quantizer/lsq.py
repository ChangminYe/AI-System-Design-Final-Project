# import torch as t
import torch
from .quantizer import Quantizer
from torch.autograd import Function
import math

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

# ## LSQ method
# class LsqQuan(Quantizer):
#     def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
#         super().__init__(bit)

#         if all_positive:
#             assert not symmetric, "Positive quantization cannot be symmetric"
#             # unsigned activation is quantized to [0, 2^b-1]
#             self.thd_neg = 0
#             self.thd_pos = 2 ** bit - 1
#         else:
#             if symmetric:
#                 # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
#                 self.thd_neg = - 2 ** (bit - 1) + 1
#                 self.thd_pos = 2 ** (bit - 1) - 1
#             else:
#                 # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
#                 self.thd_neg = - 2 ** (bit - 1)
#                 self.thd_pos = 2 ** (bit - 1) - 1

#         self.per_channel = per_channel
#         self.s = t.nn.Parameter(t.ones(1)*0.3)

#     def init_from(self, x, *args, **kwargs):
#         if self.per_channel:
#             self.s = t.nn.Parameter(
#                 x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
#         else:
#             self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

#     def forward(self, x):
#         if self.per_channel:
#             s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
#         else:
#             s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
#         s_scale = grad_scale(self.s, s_grad_scale)

#         x = x / s_scale
#         x = t.clamp(x, self.thd_neg, self.thd_pos)
#         x = round_pass(x)
#         x = x * s_scale
#         return x


####################################### Implementation of LSQ+ #######################################
## Straight through estimator
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

## Activation LSQ+
class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        a_q = Round.apply(torch.div((weight - beta), alpha)).clamp(Qn, Qp)
        a_q = a_q * alpha + beta
        return a_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller -bigger
        grad_alpha = ((smaller * Qn + bigger * Qp + 
            between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha,  None, None, None, grad_beta

## Weight LSQ+
class WLSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha 
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None

class LSQPlusWeightQuantizer(Quantizer):
    def __init__(self,  bit, all_positive=False, symmetric=False, per_channel=False):
        super().__init__(bit)
        self.w_bits = bit
        self.all_positive = all_positive
        self.batch_init = 20    ## Optimize (s, beta) for few batches(20)
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** bit - 1
        else:
            self.Qn = - 2 ** (bit - 1)
            self.Qp = 2 ** (bit - 1) - 1
        self.per_channel = per_channel
        self.init_state = 0
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, weight):
        if self.init_state==0:
            self.g = 1.0/math.sqrt(weight.numel() * self.Qp)
            self.div = 2**self.w_bits-1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data, _ = torch.max(torch.stack([torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
                self.s.data = self.s.data/self.div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data = max([torch.abs(mean-3*std), torch.abs(mean + 3*std)])/self.div
            self.init_state += 1
        elif self.init_state<self.batch_init:
            self.div = 2**self.w_bits-1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data, _ = torch.max(torch.stack([torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
                self.s.data =  self.s.data*0.9 + 0.1*self.s.data/self.div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data = self.s.data*0.9 + 0.1*max([torch.abs(mean-3*std), torch.abs(mean + 3*std)])/self.div
            self.init_state += 1
        elif self.init_state==self.batch_init:
            self.init_state += 1

        w_q = WLSQPlus.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)

        return w_q

class LSQPlusActivationQuantizer(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        self.a_bits = bit
        self.all_positive = all_positive
        self.batch_init = 20    ## Optimize (s, beta) for few batches(20)
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** bit - 1
        else:
            self.Qn = - 2 ** (bit - 1)
            self.Qp = 2 ** (bit - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(0), requires_grad=True)
        self.init_state = 0

    def forward(self, activation):
        if self.init_state==0:
            self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
            mina = torch.min(activation.detach())
            self.s.data = (torch.max(activation.detach()) - mina)/(self.Qp-self.Qn)
            self.beta.data = mina - self.s.data *self.Qn
            self.init_state += 1
        elif self.init_state<self.batch_init:
            mina = torch.min(activation.detach())
            self.s.data = self.s.data*0.9 + 0.1*(torch.max(activation.detach()) - mina)/(self.Qp-self.Qn)
            self.beta.data = self.s.data*0.9 + 0.1* (mina - self.s.data * self.Qn)
            self.init_state += 1
        elif self.init_state==self.batch_init:
            self.init_state += 1

        q_a = ALSQPlus.apply(activation, self.s, self.g, self.Qn, self.Qp, self.beta)

        return q_a

class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1)*0.3)

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = torch.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = torch.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x
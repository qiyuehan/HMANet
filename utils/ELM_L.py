import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

np.random.seed(0)
torch.manual_seed(0)

def to_var(x, requires_grad=False):
    """
    Automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.mask = None

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

class ELM_L(nn.Module):
  def __init__(self, in_dim, n_hidden):
      super(ELM_L, self).__init__()
      self.linear = MaskedLinear(in_dim, n_hidden, bias=True)
      self.dropout = nn.Dropout(0)
      self.sig = nn.Sigmoid()

      # self.beta = torch.zeros([n_hidden, out_dim])


  def initialize(self):  # 初始化模型参数
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight.data)

  def forward_L(self, x):
      x = self.linear(x)
      x = self.dropout(x)
      x = self.sig(x)
      return x

  def set_masks(self, masks):
      # Should be a less manual way to set masks
      # Leave it for the future
      self.linear.set_mask(masks[0])


  def train(self, features, targets,L=0.1):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    # assert features.shape[0] == targets.shape[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H_train = self.forward_L(features)
    L = float(0.1)  # 正则化因子(Traffic: n=500(292,298);n=600(288,296);n=1000(284, 294)_)
    self.P = torch.pinverse(torch.matmul(H_train.T, H_train)+torch.eye(H_train.shape[1]).to(device) / L)
    self.pinv = torch.matmul(self.P, H_train.T)
    self.beta = torch.matmul(self.pinv, targets)

    return self.beta


  def upgrade_beta(self, valid, target, C):
      batch_size = valid.shape[0]
      H = self.forward_L(valid)
      I = torch.eye(batch_size)/float(C)
      temp = torch.linalg.pinv(I + torch.matmul(torch.matmul(H, self.P), H.T))
      self.P -= torch.matmul(torch.matmul(torch.matmul(self.P, H.T), temp), torch.matmul(H, self.P))
      pHT = torch.matmul(self.P, H.T)
      Hbeta = torch.matmul(H, self.beta)
      self.beta += torch.matmul(pHT, target - Hbeta)


  def predict(self, features, beta):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
    H = self.forward_L(features)
    prediction = torch.matmul(H, beta)
    return prediction


  def get_beta(self,pinv, targets):
        beta = torch.matmul(pinv, targets)
        return beta

  def forecasting(self, features, beta):
    H = self.forward_L(features)
    prediction = torch.matmul(H, beta)
    return prediction
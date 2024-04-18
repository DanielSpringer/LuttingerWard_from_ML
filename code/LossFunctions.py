class DiffLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        
    def forward(self,pred,targets):    
        dfdx = vmap(grad(f), in_dims=(0, None))

class ScaledLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss
        self.eps  = eps

    def forward(self,pred,targets):
        dist_re = torch.clamp(torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values -
                    torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        dist_im = torch.clamp(torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values - 
                    torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        loss_re = self.loss(pred[:,:self.ylen] / dist_re, targets[:,:self.ylen] / dist_re)
        loss_im = self.loss(pred[:,self.ylen:] / dist_im, targets[:,self.ylen:] / dist_im)
        return loss_re + loss_im

class WeightedLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss()):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss

    def forward(self,pred,targets):

        dist_re = self.dist(
                        torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values, 
                        torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values)
        dist_im = self.dist(
                        torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values, 
                        torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values)
        scale_re = dist_im / (dist_re + dist_im)
        scale_im = dist_re / (dist_re + dist_im)
        loss_re = self.loss(scale_re * pred[:,:self.ylen], scale_re * targets[:,:self.ylen])
        loss_im = self.loss(scale_im * pred[:,self.ylen:], scale_im * targets[:,self.ylen:])
        return loss_re + loss_im
    
class WeightedLoss2(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss()):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss

    def forward(self,pred,targets):

        dist_re = self.dist(
                        torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values, 
                        torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values)
        dist_im = self.dist(
                        torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values, 
                        torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values)
        scale_re = dist_re / (dist_re + dist_im)
        scale_im = dist_im / (dist_re + dist_im)
        loss_re = self.loss(scale_re * pred[:,:self.ylen], scale_re * targets[:,:self.ylen])
        loss_im = self.loss(scale_im * pred[:,self.ylen:], scale_im * targets[:,self.ylen:])
        return loss_re + loss_im
    
class WeightedScaledLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss
        self.eps  = eps

    def forward(self,pred,targets):

        dist_re = torch.clamp(torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values -
                    torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        dist_im = torch.clamp(torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values - 
                    torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        scale_re = dist_im / (dist_re*(dist_re + dist_im))
        scale_im = dist_re / (dist_im*(dist_re + dist_im))
        loss_re = self.loss(scale_re * pred[:,:self.ylen], scale_re * targets[:,:self.ylen])
        loss_im = self.loss(scale_im * pred[:,self.ylen:], scale_im * targets[:,self.ylen:])
        return loss_re + loss_im
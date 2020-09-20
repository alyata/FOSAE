from torch import nn
import torch
import numpy as np

def pairwiseSquareLoss(a, b):
    loss = torch.pow(a.unsqueeze(1) - b, 2)
    loss = torch.sum(loss, (2,))
    loss,_ = torch.min(loss, 1)
    loss = torch.sum(loss, (0,))
    loss = torch.tensor(1/a.shape[0]) * loss
    return loss

class FOSAE(nn.Module):
    def __init__(self, objects, object_size, arity, preds, pred_units):
        super(FOSAE, self).__init__()
        self.objects = objects
        self.object_size = object_size
        self.arity = arity
        self.preds = preds
        self.pred_units = pred_units

        self.encoder = Encoder(objects, object_size, arity, preds, pred_units)
        self.decoder = nn.Sequential(
            nn.Linear(pred_units * preds * (2), #arity * object_size +
                      pred_units * preds * 4),
            nn.ReLU(),
            nn.Linear(pred_units * preds * 4, pred_units * preds * 4),
            nn.ReLU(),
            nn.Linear(pred_units * preds * 4, objects * object_size)
        )

    def forward(self, input):
        encoded = self.encoder(input)
        encoded = encoded.flatten()
        decoded =  self.decoder(encoded).reshape(self.objects, self.object_size)

        # classes = torch.nn.functional.gumbel_softmax(decoded[:,:9], hard=True)
        # classes = torch.nn.functional.pad(classes, (0, 6))
        # filter_classes = torch.tensor(9 * [[True] * 9 + [False] * 6])
        # decoded = torch.where(filter_classes, classes, decoded)
        #
        # rows = torch.nn.functional.gumbel_softmax(decoded[:,9:12], hard=True)
        # rows = torch.nn.functional.pad(rows, (9, 3))
        # filter_rows = torch.tensor(9 * [[False] * 9 + [True] * 3 + [False] * 3])
        # decoded = torch.where(filter_rows, rows, decoded)
        #
        # cols = torch.nn.functional.gumbel_softmax(decoded[:,12:15], hard=True)
        # cols = torch.nn.functional.pad(cols, (12, 0))
        # filter_cols = torch.tensor(9 * [[False] * 12 + [True] * 3])
        # decoded = torch.where(filter_cols, cols, decoded)
        return decoded

class Encoder(nn.Module):
    def __init__(self, objects, object_size, arity, preds, pred_units):
        super(Encoder, self).__init__()
        self.objects = objects
        self.object_size = object_size
        self.arity = arity
        self.preds = preds
        self.pred_units = pred_units

        self.att_units = []
        for _ in range(pred_units):
            attention_unit = nn.Sequential(
                nn.Linear(objects * object_size, 2 * objects * object_size),
                nn.ReLU(),
                nn.Linear(2 * objects * object_size, 2 * objects * object_size),
                nn.ReLU(),
                nn.Linear(2 * objects * object_size, arity * objects),
                GSMax(hard=True))
            self.att_units.append(attention_unit)

        self.pred_networks = []
        for _ in range(preds):
            pred_net = nn.Sequential(
                nn.Linear(arity * object_size, divmod(arity * object_size, 2)[0]),
                nn.ReLU(),
                nn.Linear(divmod(arity * object_size, 2)[0], divmod(arity * object_size, 2)[0]),
                nn.ReLU(),
                nn.Linear(divmod(arity * object_size, 2)[0], 2),
                GSMax(hard=True))
            self.pred_networks.append(pred_net)

    def forward(self, input):
        flat_input = input.flatten()

        attentions = [att(flat_input) for att in self.att_units]
        attentions = torch.stack(attentions, dim = 0)
        attentions = attentions.reshape(self.pred_units,
                                        self.arity,
                                        self.objects)

        pred_args = torch.matmul(attentions, input)
        pred_args = pred_args.reshape(self.pred_units,
                                      self.arity * self.object_size)

        result = [pred_net(pred_args) for pred_net in self.pred_networks]
        result = torch.stack(result, dim = 1)
        # duplicate along dimension 1 to match shape of result
        #duplicated_args = torch.cat([pred_args.unsqueeze(1)]*self.preds, 1)
        # concat the PN output to the arguments to obtain the FOL representation
        #result = torch.cat((duplicated_args, result), 2)
        return result

class GSMax(nn.Module):
    def __init__(self, tau=1, hard=False, dim=-1, eps=1e-10):
        super(GSMax, self).__init__()
        self.tau = tau
        self.hard = hard
        self.eps = eps
        self.dim = dim

    def forward(self, input):
        return nn.functional.gumbel_softmax(
            input,
            tau = self.tau,
            hard = self.hard,
            eps = self.eps,
            dim = self.dim)

def model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

import torch 
import tqdm
from torch.utils.data import DataLoader
from backpack import extend


def get_metric_function(metric_name):
    if metric_name == "logistic_accuracy":
        return logistic_accuracy

    if metric_name == "softmax_accuracy":
        return softmax_accuracy

    elif metric_name == "softmax_loss":
        return softmax_loss

    elif metric_name == "logistic_l2_loss":
        return logistic_l2_loss
    
    elif metric_name == "squared_hinge_l2_loss":
        return squared_hinge_l2_loss

    elif metric_name == "squared_l2_loss":
        return squared_l2_loss

    elif metric_name == 'perplexity':
        return lambda *args, **kwargs: torch.exp(softmax_loss(*args, **kwargs)) 
 
    elif metric_name == "logistic_loss":
        return logistic_loss

    elif metric_name == "squared_hinge_loss":
        return squared_hinge_loss

    elif metric_name == "mse":
        return mse_score

    elif metric_name == "squared_loss":
        return squared_loss


@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_name, batch_size=128):
    device = next(model.parameters()).device
    metric_function = get_metric_function(metric_name)
    
    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=batch_size)
    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    for batch in tqdm.tqdm(loader):
        images, labels = batch["images"].to(device=device), batch["labels"].to(device=device)

        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score

def logistic_l2_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    loss = criterion(logits.view(-1), labels.view(-1))

    w = 0.
    for p in model.parameters():
        w += (p**2).sum()

    loss += 1e-4 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss
    
def softmax_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    if backpack:
        criterion = extend(criterion)
    loss = criterion(logits, labels.long().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    if backpack:
        criterion = extend(criterion)
    loss = criterion(logits.view(-1), labels.float().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def squared_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.MSELoss(reduction=reduction)
    if backpack:
        criterion = extend(criterion)
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def mse_score(model, images, labels):
    logits = model(images).view(-1)
    mse = ((logits - labels.view(-1))**2).float().mean()

    return mse

def squared_hinge_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    margin=1.
    logits = model(images).view(-1)

    y = 2*labels - 1

    loss = (torch.max( torch.zeros_like(y) , 
                torch.ones_like(y) - torch.mul(y, logits)))**2

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    
    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def add_l2(model):
    w = 0.
    for p in model.parameters():
        w += (p**2).sum()

    loss = 1e-4 * w

    return loss

def squared_hinge_l2_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    loss = squared_hinge_loss(model, images, labels, backwards=False, reduction=reduction)
    loss += add_l2(model)

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def squared_l2_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    loss = squared_loss(model, images, labels, backwards=False, reduction=reduction)
    loss += add_l2(model)
    
    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_accuracy(model, images, labels):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc
import math

# learning rate decay scheduler (cosine with warmup) -- Thanks Andrej Karpathy
def get_lr_func(epoch, 
           warmup_epoch, 
           lr_decay_epoch, 
           max_lr, 
           min_lr):

    # 1) linear warmup for warmup_epoch steps
    if epoch < warmup_epoch:
        return max_lr * epoch / warmup_epoch
    
    # 2) if epoch > lr_decay_epoch, return min learning rate
    
    if epoch > lr_decay_epoch + warmup_epoch:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (epoch - warmup_epoch) / (lr_decay_epoch)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1

    return min_lr + coeff * (max_lr - min_lr)
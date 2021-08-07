import torch
import torch.nn.functional as F


def gram_lap_1d(X: torch.tensor, Y: torch.tensor, sigma=1):
    '''
    X, Y tensors of shape P, 1, d
    '''
    n_x = X.size(0)
    n_y = Y.size(0)

    d = X.size(2)
    assert Y.size(2) == d, 'data have different dimension!'

    Xlift = X.reshape(1, n_x, d)
    Ylift = Y.reshape(1, n_y, d)

    kernel = torch.exp( -torch.cdist(Xlift, Ylift, p=2) / sigma)
    
    return kernel.squeeze()


def gram_llap_1d(X: torch.tensor, Y: torch.tensor, filtersize, stride=1, sigma=1, pbc=False):
    '''
    X, Y tensors of shape P, 1, d
    '''
    n_x = X.size(0)
    n_y = Y.size(0)

    d = X.size(2)
    assert Y.size(2) == d, 'data have different dimension!'

    if pbc:
        X = F.pad(X, (0, filtersize-1), mode='circular')
        Y = F.pad(Y, (0, filtersize-1), mode='circular')
        d += filtersize-1

    Xpatch = F.unfold(X.reshape(n_x, 1, 1, d), kernel_size=(1,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    Ypatch = F.unfold(Y.reshape(n_y, 1, 1, d), kernel_size=(1,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    np = Xpatch.size(1)

    gram = torch.zeros( n_x, n_y, dtype=X.dtype, device=X.device)
    for i in range(np):
        gram_tmp = torch.exp( -torch.cdist(Xpatch[:,i,:].reshape(1, n_x, -1), Ypatch[:,i,:].reshape(1, n_y, -1) , p=2) / sigma) / ( np)
        gram.add_(gram_tmp.squeeze())

    return gram


def gram_llap_2d(X: torch.tensor, Y: torch.tensor, filtersize, stride=1, sigma=1, pbc=False):
    '''
    X, Y tensors of shape P, 1, d, d
    '''
    n_x = X.size(0)
    n_y = Y.size(0)

    d = X.size(2)
    assert Y.size(2) == d, 'data have different dimension!'
    assert X.size(3) == d, 'input not a square!'
    assert Y.size(3) == d, 'data have different dimension!'

    if pbc:
        X = F.pad(X, (0, filtersize-1, 0, filtersize-1), mode='circular')
        Y = F.pad(Y, (0, filtersize-1, 0, filtersize-1), mode='circular')
        d += filtersize-1

    Xpatch = F.unfold(X.reshape(n_x, 1, d, d), kernel_size=(filtersize,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    Ypatch = F.unfold(Y.reshape(n_y, 1, d, d), kernel_size=(filtersize,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    np = Xpatch.size(1)

    gram = torch.zeros( n_x, n_y, dtype=X.dtype, device=X.device)
    for i in range(np):
        gram_tmp = torch.exp( -torch.cdist(Xpatch[:,i,:].reshape(1, n_x, -1), Ypatch[:,i,:].reshape(1, n_y, -1) , p=2) / sigma) / ( np)
        gram.add_(gram_tmp.squeeze())

    return gram


def gram_clap_1d(X: torch.tensor, Y: torch.tensor, filtersize, stride=1, sigma=1, pbc=False):
    '''
    X,Y tensors of shape (P,1,d)
    '''
    n_x = X.size(0)
    n_y = Y.size(0)

    d = X.size(2)
    assert Y.size(2) == d, 'data have different dimension!'

    if pbc:
        X = F.pad(X, (0, filtersize-1), mode='circular')
        Y = F.pad(Y, (0, filtersize-1), mode='circular')
        d += filtersize-1

    Xpatch = F.unfold(X.reshape(n_x, 1, 1, d), kernel_size=(1,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    Ypatch = F.unfold(Y.reshape(n_y, 1, 1, d), kernel_size=(1,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    np = Xpatch.size(1)

    gram = torch.zeros( n_x, n_y, dtype=X.dtype, device=X.device)
    for i in range(np):
        for j in range(np):
            gram_tmp = torch.exp( -torch.cdist(Xpatch[:,i,:].reshape(1, n_x, -1), Ypatch[:,j,:].reshape(1, n_y, -1) , p=2) / sigma) / ( np * np)
            gram.add_(gram_tmp.squeeze())

    return gram

def gram_clap_2d(X: torch.tensor, Y: torch.tensor, filtersize, stride=1, sigma=1, pbc=False):
    '''
    X, Y tensors of shape P, 1, d, d
    '''
    n_x = X.size(0)
    n_y = Y.size(0)

    d = X.size(2)
    assert Y.size(2) == d, 'data have different dimension!'
    assert X.size(3) == d, 'input not a square!'
    assert Y.size(3) == d, 'data have different dimension!'

    if pbc:
        X = F.pad(X, (0, filtersize-1, 0, filtersize-1), mode='circular')
        Y = F.pad(Y, (0, filtersize-1, 0, filtersize-1), mode='circular')
        d += filtersize-1

    Xpatch = F.unfold(X.reshape(n_x, 1, d, d), kernel_size=(filtersize,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    Ypatch = F.unfold(Y.reshape(n_y, 1, d, d), kernel_size=(filtersize,filtersize), dilation=1, padding=0, stride=stride).transpose(1,2)
    np = Xpatch.size(1)

    gram = torch.zeros( n_x, n_y, dtype=X.dtype, device=X.device)
    for i in range(np):
        for j in range(np):
            gram_tmp = torch.exp( -torch.cdist(Xpatch[:,i,:].reshape(1, n_x, -1), Ypatch[:,j,:].reshape(1, n_y, -1) , p=2) / sigma) / ( np)
            gram.add_(gram_tmp.squeeze())

    return gram


"""
Computes the Gram matrix for the NTK of a given model f
"""
def compute_kernels(f, xtr, xte, parameters=None):
    from hessian import gradient

    if parameters is None:
        parameters = list(f.parameters())

    ktrtr = xtr.new_zeros(len(xtr), len(xtr))
    ktetr = xtr.new_zeros(len(xte), len(xtr))
    ktete = xtr.new_zeros(len(xte), len(xte))

    params = []
    current = []
    for p in sorted(parameters, key=lambda p: p.numel(), reverse=True):
        current.append(p)
        if sum(p.numel() for p in current) > 2e9 // (8 * (len(xtr) + len(xte))):
            if len(current) > 1:
                params.append(current[:-1])
                current = current[-1:]
            else:
                params.append(current)
                current = []
    if len(current) > 0:
        params.append(current)

    for i, p in enumerate(params):
        print("[{}/{}] [len={} numel={}]".format(i, len(params), len(p), sum(x.numel() for x in p)), flush=True)

        jtr = xtr.new_empty(len(xtr), sum(u.numel() for u in p))  # (P, N~)
        jte = xte.new_empty(len(xte), sum(u.numel() for u in p))  # (P, N~)

        for j, x in enumerate(xtr):
            jtr[j] = gradient(f(x[None]), p)  # (N~)

        for j, x in enumerate(xte):
            jte[j] = gradient(f(x[None]), p)  # (N~)

        ktrtr.add_(jtr @ jtr.t())
        ktetr.add_(jte @ jtr.t())
        ktete.add_(jte @ jte.t())
        del jtr, jte

    return ktrtr, ktetr, ktete

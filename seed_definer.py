import os, torch, random
import numpy as np

def set_random_seed(seed):    
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)    
    random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)
    #torch.use_deterministic_algorithms(True)
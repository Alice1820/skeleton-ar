from __future__ import print_function

import torch
import numpy as np

idx2label = {
  'ntu': {
    1: 'drink water',   
    2: 'eat meal',   
    3: 'brush teeth',  
    4: 'brush hair',  
    5: 'drop',
    6: 'pick up', 
    7: 'throw ',   
    8: 'sit down ',   
    9: 'stand up ',   
    10: 'clapping    ',
    11: 'reading  ',  
    12: 'writing   ', 
    13: 'tear up paper   ', 
    14: 'put on jacket    ',
    15: 'take off jacket   ', 
    16: 'put on a shoe   ', 
    17: 'take off a shoe    ',
    18: 'put on glasses ',   
    19: 'take off glasses    ',
    20: 'put on a hat/cap ',   
    21: 'take off a hat/cap    ',
    22: 'cheer up   ', 
    23: 'hand waving   ', 
    24: 'kicking something   ', 
    25: 'reach into pocket  ',  
    26: 'hopping   ', 
    27: 'jump up ',   
    28: 'phone call',    
    29: 'play with phone/tablet',    
    30: 'type on a keyboard  ',  
    31: 'point to something',    
    32: 'taking a selfie    ',
    33: 'check time (from watch)  ',  
    34: 'rub two hands',    
    35: 'nod head/bow  ',  
    36: 'shake head    ',
    37: 'wipe face  ',  
    38: 'salute  ',  
    39: 'put palms together',    
    40: 'cross hands in front   ', 
    61: 'put on headphone ',   
    62: 'take off headphone   ', 
    63: 'shoot at basket  ',  
    64: 'bounce ball   ', 
    65: 'tennis bat swing  ',  
    66: 'juggle table tennis ball   ', 
    67: 'hush',
    68: 'flick hair'   ,
    69: 'thumb up    ',
    70: 'thumb down    ',
    71: 'make OK sign ',   
    72: 'make victory sign   ', 
    73: 'staple book',    
    74: 'counting money    ',
    75: 'cutting nails    ',
    76: 'cutting paper',    
    77: 'snap fingers ',   
    78: 'open bottle ',   
    79: 'sniff/smell ',   
    80: 'squat down   ', 
    81: 'toss a coin    ',
    82: 'fold paper  ',  
    83: 'ball up paper ',   
    84: 'play magic cube ',   
    85: 'apply cream on face',    
    86: 'apply cream on hand',    
    87: 'put on bag    ',
    88: 'take off bag    ',
    89: 'put object into bag',    
    90: 'take object out of bag',    
    91: 'open a box   ', 
    92: 'move heavy objects',    
    93: 'shake fist   ', 
    94: 'throw up cap/hat',    
    95: 'capitulate   ', 
    96: 'cross arms   ', 
    97: 'arm circles   ', 
    98: 'arm swings   ', 
    99: 'run on the spot',    
    100: 'butt kicks   ', 
    101: 'cross toe touch',    
    102: 'side kick   ', 
    41: 'sneeze/cough',
    42: 'staggering',
    43: 'falling down',
    44: 'headache',
    45: 'chest pain',
    46: 'back pain',
    47: 'neck pain',
    48: 'nausea/vomiting',
    49: 'fan self',
    103: 'yawn',
    104: 'stretch oneself',
    105: 'blow nose',
    50: 'punch/slap',
    51: 'kicking',
    52: 'pushing',
    53: 'pat on back',
    54: 'point finger',
    55: 'hugging',
    56: 'giving object',
    57: 'touch pocket',
    58: 'shaking hands',
    59: 'walking towards',
    60: 'walking apart',
    106: 'hit with object',
    107: 'wield knife',
    108: 'knock over',
    109: 'grab stuff',
    110: 'shoot with gun',
    111: 'step on foot',
    112: 'high-five',
    113: 'cheers and drink',
    114: 'carry object',
    115: 'take a photo',
    116: 'follow',
    117: 'whisper',
    118: 'exchange things',
    119: 'support somebody',
    120: 'rock-paper-scissors',
    },
    'fall': {
    1: 'lying',
    2: 'falling',
    3: 'background',
    },
    'fallv3':
    {
    1: 'background',
    2: 'falling',
    }
}

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print (correct.size())
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    meter = AverageMeter()

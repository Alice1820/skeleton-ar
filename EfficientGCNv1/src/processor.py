import logging, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from time import time
import datetime

import threading

import cv2

from . import utils as U
from .initializer import Initializer

idx2label = {
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
}

class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            print (x.size())
            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
        ))
        logging.info('')

    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                
                print (x.size())
                # print (x[0, 0, 0, 0, :, 0])
                # Calculating Output
                out, _ = self.model(x)
                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out,5)[1]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm

    def demo(self):
        self.model.eval()
        start_eval_time = time()
        # feed clip into model
        threading.Thread(target=self.demo_batch, daemon=True).start()
        k = 0
        while True:
            if self.bodyTracker.imageNow is not None:
                # Overlay body segmentation on depth image
                cv2.imshow('Segmented Depth Image', self.bodyTracker.imageNow)
                k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('q'):
                cv2.imwrite('outputImage.jpg', self.bodyTracker.imageNow)

    def demo_batch(self):
        # interest = torch.tensor([[22], [26]])
        # y_onehot = torch.zeros([2, 120])
        # y_onehot.scatter_(1, interest, 1)
        # y_onehot = torch.sum(y_onehot, dim=0, keepdim=True) # [1, 120]
        # y_onehot = y_onehot.cuda()
        # print (y_onehot)
        # idx2label = {23: 'hand waving', 27: 'jumping up'}
        # idx2label = {0: 'hand waving', 1: 'jumping up'}
        # out_i = torch.zeros([1, 2])
        self.T = 288
        while True:
            x = np.zeros([3, 6, self.T, 25, 2]).astype(np.float32)
            with torch.no_grad():
                ske = self.bodyTracker.next_skeleton()
                if ske is not None:
                    x[:, :, :32, :, :] = np.copy(ske)
                    x = torch.Tensor(x).float()
                    x = x.unsqueeze(0).to(self.device)
                    # Calculating Output
                    # print (x.size())
                    # print (x[0, 0, 0, 0, :, 0])
                    out, _ = self.model(x) # [bs, 120]
                    out = F.softmax(out, -1)
                    # out = out * y_onehot
                    # mapping
                    # out_i[:, 0] = out[:, 22]
                    # out_i[:, 1] = out[:, 26]
                    # out_i = out_i.data.cpu()
                    # out_i = F.softmax(out_i, -1)
                    # out = F.softmax(out, -1)
                    # print (out)
                    
                    prob = torch.max(out[0]).cpu().numpy()
                    pred = torch.argmax(out[0]).cpu().numpy()
                    out = out.cpu().numpy()
                    ind = np.argpartition(out[0], -5)[-5:]
                    # print out result
                    if prob > 0.5:
                        print (datetime.datetime.now().__str__() + '         ', '{:.2f}%'.format(prob * 1e2), '{:03d}'.format(pred + 1), \
                            idx2label[int(pred) + 1], [idx2label[ind[i] + 1] for i in range(5)])
                        # print (out[0].cpu().numpy())

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')
        
        elif self.args.demo:
            logging.info('Loading demo model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting testing ...')
            self.demo()
        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=self.cm,
                feature=feature, weight=weight, location=location
            )
        logging.info('Finish extracting!')
        logging.info('')

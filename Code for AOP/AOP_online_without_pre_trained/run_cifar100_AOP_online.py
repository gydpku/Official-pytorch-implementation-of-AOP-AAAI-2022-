import sys, argparse
import numpy as np
import torch
import utils
import datetime
import  pdb
# Arguments
parser=argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default= 0, help='(default=%(default)d)')
parser.add_argument('--experiment',default='cifar-10', type=str,required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='OWM', type=str, required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs', default=1, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.04, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar10', 'cifar10ori', 'cifar100ori', 'cifar100', 'cifar50', 'emnist',
                             'twentynews', 'dbpedia'], help='gpu no.')
parser.add_argument('--gpu', type=int, default=0, help='gpu no.')
parser.add_argument('--temperature', help='Temperature for similarity',
                    default=0.5, type=float)
parser.add_argument('--mode', help='Training mode',
                    default='simclr', type=str)
parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                    default=128, type=int)
parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                    choices=['rotation', 'cutperm', 'none'], type=str)
parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                    default=0.08, type=float)
args = parser.parse_args()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use gpu0,1

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args,arg))
print('='*100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)
########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()
import cifar as dataloader
import AOP as approach
import cnn_owm as network

########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get_cifar100(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)
# pdb.set_trace()
# Inits
print('Inits...')
num_class=100
net = network.Net(inputsize,num_class=num_class).cuda()
utils.print_model_report(net)

appr = approach.Appr(model=net, nepochs=args.nepochs,sbatch=32, lr=args.lr, args=args)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['test']['x'].cuda()
    yvalid = data[t]['test']['y'].cuda()

    # Train
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data,num_class=num_class,test_id=10)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(xtest, ytest,num_class=num_class)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss


# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]),end='')
    print()
print('*'*100)
print('Done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)





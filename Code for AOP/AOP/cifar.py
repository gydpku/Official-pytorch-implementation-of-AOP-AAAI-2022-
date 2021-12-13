import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import pdb


def get(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar10_020/'):
        os.makedirs('./data/binary_cifar10_020')
        t_num = 1
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar10_020'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar10_020'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar10_020'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar10_020'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:10//data[0]['ncla']], size

def get2(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar10_01/'):
        os.makedirs('./data/binary_cifar10_01')
        t_num = 1
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar10_01'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar10_01'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar10_01'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar10_01'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:10//data[0]['ncla']], size

def get_MNIST(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # CIFAR10
    if not os.path.isdir('/home/guoyd/Dataset/MNIST4/'):
        os.makedirs('/home/guoyd/Dataset/MNIST4')
        t_num = 1
        mean = [x / 255 for x in [0.1307, ]]
        std = [x / 255 for x in [0.3081,]]
        dat={}
        dat['train']=datasets.MNIST('./data2/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
        dat['test']=datasets.MNIST('./data2/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

        for t in range(10//t_num):
            print("ttt",t)
            data[t] = {}
            data[t]['name'] = 'mnist-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    print("t",target)
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'mnist-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'MNIST->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:10//data[0]['ncla']], size

def get_cifar100(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_100/'):
        os.makedirs('./data/binary_cifar100_100')
        t_num = 1
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_100'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_100'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(101))
    print('Task order =', ids)
    for i in range(101):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_100'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_100'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:100//data[0]['ncla']], size

def get_imagenet(seed=0,pc_valid=0.10,download=False):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_imagenet_200/'):
        os.makedirs('./data/binary_imagenet_200')
        t_num = 1
        mean = [x / 255 for x in [0.48, 0.448, 0.3975]]
        std = [x / 255 for x in [0.277, 0.2691, 0.2821]]
        #download=True
        root='./home/guoyd/'
        if download:
            #if os.path.isdir(root) and len(os.listdir(root)) > 0:
             #   print('Download not needed, files already on disk.')
            #else:
            from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        test_data = []
        train_data = []
        for num in range(20):
            train_data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train', num+1))))
            test_data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('val', num + 1))))
        train_data = np.concatenate(np.array(train_data))
        test_data = np.concatenate(np.array(test_data))

        train_targets = []
        test_targets = []
        for num in range(20):
            train_targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' , num+1))))
            test_targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('val' , num+1))))
        train_targets = np.concatenate(np.array(train_targets))
        test_targets = np.concatenate(np.array(test_targets))
        transform_data = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform_target = transforms.Compose([transforms.ToTensor()])
        train_data = transform_data(train_data)
        test_data = transform_data(test_data)
        train_targets = transform_target(train_targets)
        test_targets = transform_target(test_targets)
        train_D=torch.utils.data.TensorDataset(train_data,train_targets)
        test_D=torch.utils.data.TensorDataset(test_data,test_targets)

       # img = Image.fromarray(np.uint8(255 * img))
        dat={}
        dat['train']=train_D#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=test_D#datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(200//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'imagnet-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    print("label")
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 200 // t_num
        data[t] = {}
        data[t]['name'] = 'tinyimagent-all'
        data[t]['ncla'] = 200
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_imagenet_200'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_imagenet_200'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(201))
    print('Task order =', ids)
    for i in range(201):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_imagenet_200'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_imagenet_200'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'imagenet200->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:100//data[0]['ncla']], size

def get_EMNIST(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # CIFAR10

    if not os.path.isdir('/home/guoyd/Dataset/EMNIST47_1/'):
        os.makedirs('/home/guoyd/Dataset/EMNIST47_1')
        t_num = 1
        mean = [x / 255 for x in [0.1307, ]]
        std = [x / 255 for x in [0.3081,]]
        dat={}
        dat['train']=datasets.EMNIST('./data2/',split='balanced', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
        dat['test']=datasets.EMNIST('./data2/', split='balanced',train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

        for t in range(47//t_num):
            print("ttt",t)
            data[t] = {}
            data[t]['name'] = 'mnist-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                   # print("t",target)
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 47 // t_num
        data[t] = {}
        data[t]['name'] = 'Emnist-all'
        data[t]['ncla'] = 47
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_1/'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_1/'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(48))
    print('Task order =', ids)
    for i in range(48):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_1/'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_1/'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'EMNIST->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:47//data[0]['ncla']], size
def get_EMNIST_2(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # CIFAR10

    if not os.path.isdir('/home/guoyd/Dataset/EMNIST47_2/'):
        os.makedirs('/home/guoyd/Dataset/EMNIST47_2')
        t_num = 2
        mean = [x / 255 for x in [0.1307, ]]
        std = [x / 255 for x in [0.3081,]]
        dat={}
        dat['train']=datasets.EMNIST('./data2/',split='balanced', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
        dat['test']=datasets.EMNIST('./data2/', split='balanced',train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

        for t in range(47//t_num):
            print("ttt",t)
            data[t] = {}
            data[t]['name'] = 'mnist-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                   # print("t",target)
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 47 // t_num
        data[t] = {}
        data[t]['name'] = 'mnist-' + str(t_num * t) + '-' + str(t_num * (t + 1) - 1)
        data[t]['ncla'] = 1
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                # print("t",target)
                if label in range(46, 47):
                    data[t][s]['x'].append(image)
                    data[t][s]['y'].append(label)
        t = (47 // t_num) +1
        data[t] = {}
        data[t]['name'] = 'Emnist-all'
        data[t]['ncla'] = 47
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_2/'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_2/'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(25))
    print('Task order =', ids)
    for i in range(25):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_2/'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/EMNIST47_2/'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'EMNIST->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:24], size

def get_cifar100_2(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_2/'):
        os.makedirs('./data/binary_cifar100_2')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(100//t_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_2'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_2'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(51))
    print('Task order =', ids)
    for i in range(51):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_2'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar100_2'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:100//data[0]['ncla']], size

def get_tinyimagnet(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # CIFAR10
    if not os.path.isdir('/home/guoyd/Dataset/MNIST4/'):
        os.makedirs('/home/guoyd/Dataset/MNIST4')
        t_num = 1
        mean = [x / 255 for x in [0.1307, ]]
        std = [x / 255 for x in [0.3081,]]
        dat={}
        dat['train']=datasets.MNIST('./data2/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
        dat['test']=datasets.MNIST('./data2/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

        for t in range(10//t_num):
            print("ttt",t)
            data[t] = {}
            data[t]['name'] = 'mnist-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    print("t",target)
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'mnist-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('/home/guoyd/Dataset/MNIST4/'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'MNIST->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    # pdb.set_trace()

    return data, taskcla[:10//data[0]['ncla']], size

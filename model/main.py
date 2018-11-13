import torchvision
import torch
import SlideDataset
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from optparse import OptionParser
from datetime import datetime

usage = "usage: python main.py "
parser = OptionParser(usage)

parser.add_option("-l", "--learning_rate", dest="learning_rate", type="float", default=0.0001,
                    help="set learning rate for optimizor")
parser.add_option("-b", "--batch_size", dest="batch_size", type="int", default=17,
                    help="batch size")
parser.add_option("-o", "--output", dest="output", type="string", default="specified_format.csv",
                    help="output file")
parser.add_option("-r", "--resume", dest="model_file", type="string", default="",
                    help="resume the file from a model file")

(options, args) = parser.parse_args()

batch_size = options.batch_size
embed_len = 10
sample_num_epoch = 50000
epoch_num = 200

if options.model_file == "":
    # for downloading pretrained model
    inception_v3_pretn = torchvision.models.inception_v3(pretrained=True)
    pretn_state_dict = inception_v3_pretn.state_dict()
    del inception_v3_pretn

    inception_v3 = torchvision.models.inception_v3(num_classes=embed_len, aux_logits=False)
    model_state_dict = inception_v3.state_dict()

    # restore all the weight except the last layer
    update_state = {k:v for k, v in pretn_state_dict.items() if k not in ["fc.weight", "fc.bias"] and k in model_state_dict}
    model_state_dict.update(update_state)
    inception_v3.load_state_dict(model_state_dict)

    linear_em_2 = torch.nn.Linear(embed_len, 2)
    model_site = torch.nn.Sequential(
                torch.nn.BatchNorm1d(embed_len),
                torch.nn.ReLU(True),
                linear_em_2,
            )

    inception_v3 = torch.nn.DataParallel(inception_v3).cuda()
    model_site = torch.nn.DataParallel(model_site).cuda()
else:
    '''
    define your own restoring function here
    '''
    model_site = torch.load("model_site.pt")
    inception_v3 = torch.load("inception_tran.pt")

learning_rate = options.learning_rate

optimizer = torch.optim.Adam([
            {'params': model_site.parameters()},
            {'params': inception_v3.parameters()}
        ],
        lr=learning_rate)

loss_fn_site = torch.nn.CrossEntropyLoss().cuda()

'''
define your own transformer here
'''
train_transform = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor()
            ])

val_transform = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor()
            ])

train_image_data = SlideDataset.SlideDataset('../data/Tile/val/COAD/', '../data/Tile/val/UCEC/', train_transform )
train_data_loader = torch.utils.data.DataLoader(train_image_data, batch_size=batch_size, shuffle=False)

val_image_data = SlideDataset.SlideDataset('../data/Tile/val/COAD/', '../data/Tile/val/UCEC/', val_transform )
val_data_loader = torch.utils.data.DataLoader(val_image_data, batch_size=batch_size, shuffle=False)

def get_acc(pred, var):
    max_value, index = torch.max(pred, 1)
    index = index.data.cpu().numpy()
    var = var.data.cpu().numpy()
    return np.sum(index == var)*1.0/index.shape[0]

writer = SummaryWriter(log_dir="runs/baseline_" + datetime.now().strftime('%b%d_%H-%M-%S'))
step = 0

for epoch in range(epoch_num):
    print("Epoch: " + str(epoch))
    
    sum_y_acc = 0.0
    sum_loss = 0.0
    count = 0
    
    inception_v3.train()
    model_site.train()

    for id, (item, y) in enumerate(train_data_loader):
        item = item.cuda()
        input_var = torch.autograd.Variable(item, requires_grad=True)
        
        y = y.cuda()
        y = torch.transpose(y, 0, 1)
        y = torch.squeeze(y)
        y_var = torch.autograd.Variable(y)

        optimizer.zero_grad()
        
        res = inception_v3(input_var)
        y_pred = model_site(res)
        
        loss_site = loss_fn_site(y_pred, y_var)
        
        loss_site.backward()
        optimizer.step()

        cur_site_loss = loss_site.data.cpu().numpy()
        print( "Training Epoch: " + str(epoch) + ", id: " + str(id) + ", classifier_loss: " + str(cur_site_loss))
        
        step += 1
        writer.add_scalar('step/site_loss', cur_site_loss, step)
        
        sum_loss += cur_site_loss
        count += 1
        cur_y_acc = get_acc(y_pred, y_var)

        writer.add_scalar('step/site_acc', cur_y_acc, step)
 
        sum_y_acc += cur_y_acc
    
    print( "training site average accuracy: ")
    print( sum_y_acc / count)
    writer.add_scalar('epoch/train_average_acc', sum_y_acc / count, epoch)
    print( "training average loss: ")
    print( sum_loss / count)
    writer.add_scalar('epoch/train_average_loss', sum_loss / count, epoch)

    #writer.export_scalars_to_json("./inception_tran_all_scalars.json")
    
    with open("inception_tran_" + str(epoch) + ".pt", 'wb') as f: 
        torch.save(inception_v3, f)
    with open("model_site_" + str(epoch) + ".pt", 'wb') as f: 
        torch.save(model_site, f)
   
    #val 
    sum_y_acc = 0.0
    sum_loss = 0.0
    count = 0
    inception_v3.eval()
    model_site.eval()
    for id, (item, y) in enumerate(val_data_loader):
        item = item.cuda()
        input_var = torch.autograd.Variable(item, requires_grad=False)
        
        y = y.cuda()
        y = torch.transpose(y, 0, 1)
        y = torch.squeeze(y)
        y_var = torch.autograd.Variable(y)

        with torch.no_grad():
            res = inception_v3(input_var)
            y_pred = model_site(res)
            
            loss_site = loss_fn_site(y_pred, y_var)

            cur_loss = loss_site.data.cpu().numpy()
            
            print("Val Epoch: " + str(epoch) + ", id: " + str(id) + ", val_loss: " + str(cur_loss))
            sum_loss += cur_loss
            count += 1
            cur_y_acc = get_acc(y_pred, y_var)
            sum_y_acc += cur_y_acc

    print("val average accuracy: ")
    print( sum_y_acc / count)
    writer.add_scalar('epoch/val_average_acc', sum_y_acc / count, epoch)
    print("val average loss: ")
    print( sum_loss / count)
    writer.add_scalar('epoch/val_average_loss', sum_loss / count, epoch)

writer.close()

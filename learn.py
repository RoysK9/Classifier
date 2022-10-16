# import os
# import random
# import copy
# import decimal
# import csv


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import confusion_matrix


# import torch
# from torchsummary import summary
# from torch import nn,optim
# from torch.functional import split
# from torch.utils.data import DataLoader, TensorDataset, Dataset
# from torchvision import transforms
# from tqdm import *

from utils.logger import *
from utils.make_file_name import *
from utils.early_stopping import *
from utils.plot_result import *
from multi_balance_sample import *
from seed_definer import *
from model import *
from mydataset import Mydataset

from utils.import_libraries import *


def learn(param):

    model = Model(param).to(device)
    if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
        logger.info('use_multi')
        model = torch.nn.DataParallel(model) # make parallel

    optimizer = optim.Adam(model.parameters(), lr=param.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,40], gamma=0.5)
    criterion = nn.CrossEntropyLoss().to(device)

    file_name = make_file_name(param)

    os.makedirs("./weights", exist_ok=True)
    best_model_path = os.path.join('./weights', file_name + '_best_model_weight.pth') 
    last_model_path = os.path.join('./weights', file_name + '_last_model_weight.pth') 

    record_loss_train = []
    record_loss_val = []
    record_train_accuracy = [0]
    record_val_accuracy = [0]

    train_loader = MultiBalanceSampler(mode='train', n_samples=7) #trainデータに対してはmulti_balance_sampleのloader化処理を加える
    #val_loader = MultiBalanceSampler(mode='val', n_samples=1)
    
    #train_dataset = Mydataset('train')
    #train_loader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, drop_last=False, num_workers=2)
    val_dataset = Mydataset('val') #valデータに対しては通常のloader化処理を加える
    val_loader = DataLoader(val_dataset, batch_size=param.batch_size, shuffle=True, drop_last=False, num_workers=2)

    logger.info('Learning...')

    classes = ['1','2','3','4','5','6','7','8','9']
    num_class = len(classes)

    train_miss_num = 0
    train_miss_folder = './pred_miss/train_' + file_name
    os.makedirs(train_miss_folder, exist_ok=True)

    val_miss_num = 0
    val_miss_folder = './pred_miss/val_' + file_name
    os.makedirs(val_miss_folder, exist_ok=True)

    early_stopping = EarlyStopping(path=best_model_path)

    for i in range(1,param.epochs+1):
        model.train()
        loss_train = 0
        loss_val = 0
        loop = 0

        train_correct = 0
        train_total = 0
        train_class_correct = list(0. for i in range(num_class))
        train_class_total = list(0. for i in range(num_class))

        pred_label = torch.tensor([]).to(device)
        true_label = torch.tensor([]).to(device)

        for j, (x, t) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            out = model(x)

            loss = criterion(out, t)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

            loop += 1

            _, predicted = torch.max(out, 1)
            c = (predicted == t).squeeze()

            if j==1:
                pred_label = predicted
                true_label = t
            else:    
                pred_label = torch.cat((pred_label,predicted),0)
                true_label = torch.cat((true_label,t),0)

            train_total += t.size(0)
            train_correct += (predicted == t).sum().item()

            for k in range(t.shape[0]):
                label = t[k]
                train_class_correct[label] += c[k].item()
                train_class_total[label] += 1

                if i == param.epochs:

                    if t[k]!=predicted[k]:
                    
                        miss_img = x[k]

                        img = miss_img.to('cpu').detach().numpy()
                        img = np.transpose(img,(1,2,0))
                        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        label = t[k].item()
                        predict = predicted[k].item()
                        ax_pos = plt.axis()  
                        plt.text(ax_pos[0] + 5, ax_pos[2] + 5, 'Label:' + classes[label] + ' Pred:' + classes[predict], color="cyan")  
                        plt.imshow(img)
                        plt.xlim(0,64)
                        plt.ylim(64,0)

                        plt.savefig(train_miss_folder + '/' + str(train_miss_num) + '.png')

                        plt.gca().clear()

                        train_miss_num += 1
            
        loss_train /= loop
        record_loss_train.append(loss_train)
        
        train_pred_label = pred_label.to('cpu').detach().numpy().copy()
        train_true_label = true_label.to('cpu').detach().numpy().copy()

        train_confusion_matrix = confusion_matrix(train_true_label, train_pred_label)
        logger.info(train_confusion_matrix)

        train_accuracy = 100 * train_correct / train_total
        record_train_accuracy.append(train_accuracy/100)
        for j in range(num_class):
            logger.info(f'Accuracy of {classes[j]} : {train_class_correct[j] / train_class_total[j]:.1%}')

        val_correct = 0
        val_total = 0
        val_class_correct = list(0. for i in range(num_class))
        val_class_total = list(0. for i in range(num_class))
        loop = 0
        model.eval()

        for j, (x, t) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                x = x.to(device)
                t = t.to(device)
                out = model(x)

                loss = criterion(out, t)
                loss_val += loss.item()

                loop += 1

                _, predicted = torch.max(out, 1)
                c = (predicted == t).squeeze()
                
                if j==1:
                    pred_label = predicted
                    true_label = t
                else:    
                    pred_label = torch.cat((pred_label,predicted),0)
                    true_label = torch.cat((true_label,t),0)

                val_total += t.size(0)
                val_correct += (predicted == t).sum().item()

                for k in range(t.shape[0]):
                    label = t[k]
                    val_class_correct[label] += c[k].item()
                    val_class_total[label] += 1

                    if i == param.epochs:
                            
                        if t[k]!=predicted[k]:
                            
                            miss_img = x[k]

                            img = miss_img.to('cpu').detach().numpy()
                            img = np.transpose(img,(1,2,0))
                            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                            label = t[k].item()
                            predict = predicted[k].item()
                            ax_pos = plt.axis()  
                            plt.text(ax_pos[0] + 5, ax_pos[2] + 5, 'Label:' + classes[label] + ' Pred:' + classes[predict], color="cyan")  
                            plt.imshow(img)
                            plt.xlim(0,64)
                            plt.ylim(64,0)

                            plt.savefig(val_miss_folder + '/' + str(val_miss_num) + '.png')

                            plt.gca().clear()

                            val_miss_num += 1

        loss_val /= loop
        record_loss_val.append(loss_val)

        val_pred_label = pred_label.to('cpu').detach().numpy().copy()
        val_true_label = true_label.to('cpu').detach().numpy().copy()

        val_confusion_matrix = confusion_matrix(val_true_label, val_pred_label)
        logger.info(val_confusion_matrix)

        val_accuracy = 100 * val_correct / val_total
        record_val_accuracy.append(val_accuracy/100)
        for j in range(num_class):
            logger.info(f'Accuracy of {classes[j]} : {val_class_correct[j] / val_class_total[j]:.1%}')


        early_stopping(loss_val, model) # 最良モデルならモデルパラメータ保存
        if early_stopping.early_stop: 
            # 一定epochだけval_lossが最低値を更新しなかった場合、ここに入り学習を終了
            break

        logger.info(f'Epoch: {i}',)
        logger.info(f'Train Loss:{loss_train}')
        logger.info(f'Val Loss:{loss_val}')
        logger.info(f'Train_Accuracy:{train_accuracy}')
        logger.info(f'Val_Accuracy:{val_accuracy}')

    #ここからはLossやAccuracyをグラフにしたりcsvに出力したりする部分

    make_loss_graph(record_loss_train, record_loss_val, file_name)
    make_acc_graph(record_train_accuracy,record_val_accuracy,file_name)

    os.makedirs("./results", exist_ok=True)
    f = open('./results/' + file_name+'_Result.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['train_loss'])
    writer.writerow(record_loss_train)
    writer.writerow(['val_loss'])
    writer.writerow(record_loss_val)
    writer.writerow('')
    writer.writerow(['final_train_loss'])
    writer.writerow([record_loss_train[-1]])
    writer.writerow(['final_val_loss'])
    writer.writerow([record_loss_val[-1]])
    writer.writerow('')
    writer.writerow(['Train Accuracy'])
    writer.writerow([record_train_accuracy[-1]*100])
    writer.writerow(['Val Accuracy'])
    writer.writerow([record_val_accuracy[-1]*100])
    for i in range(num_class):
        writer.writerow(['Train Accuracy of ' + classes[i]])
        writer.writerow([train_class_correct[i]/train_class_total[i]*100])
    for i in range(num_class):
        writer.writerow(['Val Accuracy of ' + classes[i]])
        writer.writerow([val_class_correct[i] / val_class_total[i]*100])
    writer.writerow(['Train Confusion Matrix'])
    for i in range(num_class):
        writer.writerow([train_confusion_matrix[i][:]])
    writer.writerow(['Val Confusion Matrix'])
    for i in range(num_class):
        writer.writerow([val_confusion_matrix[i][:]])
    f.close()

    torch.save(model.state_dict(), last_model_path) 
    print(model.fc1.weight)
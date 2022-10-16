from utils.logger import *
from utils.make_file_name import *
from utils.early_stopping import *
from utils.plot_result import *
from multi_balance_sample import *
from seed_definer import *
from model import *
from mydataset import Mydataset
from utils.parameter_loader import *

from utils.import_libraries import *


def test(param):

    file_name = make_file_name(param)
    os.makedirs("./weights", exist_ok=True)
    best_model_path = os.path.join('./weights', file_name + '_best_model_weight.pth') 
    last_model_path = os.path.join('./weights', file_name + '_last_model_weight.pth') 

    test_data = Mydataset('test')
    test_loader = DataLoader(test_data, batch_size=param.batch_size, shuffle=False, drop_last=False, num_workers=2)

    classes = ['1','2','3','4','5','6','7','8','9']

    num_class = len(classes)

    correct = 0
    total = 0

    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))

    pred_label = torch.tensor([]).to(device)
    true_label = torch.tensor([]).to(device)

    miss_num = 0
    miss_folder = './pred_miss/test_' + file_name
    os.makedirs(miss_folder, exist_ok=True)

    correct_folder = './pred_correct/test_' + file_name

    model = Model(param).to(device)
    model.load_state_dict(torch.load(last_model_path))
    print(model.fc1.weight)

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            if i==0:
                pred_label = predicted
                true_label = labels
            else:    
                pred_label = torch.cat((pred_label,predicted),0)
                true_label = torch.cat((true_label,labels),0)

            total += labels.size(0)

            for j in range(labels.shape[0]):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1

                if (labels[j] == predicted[j]):

                    if(True):   # 全画像出力がしたいならTrueにする

                        os.makedirs(correct_folder, exist_ok=True)
                        img = inputs[j]
                        img = img.to('cpu').detach().numpy()
                        img = np.transpose(img,(1,2,0))
                        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        label = labels[j].item()
                        predict = predicted[j].item()

                        ax_pos = plt.axis()  
                        plt.text(ax_pos[0] + 5, ax_pos[2] + 5, 'Label:' + classes[label] + ' Pred:' + classes[predict], color="cyan") 

                        plt.imshow(img)
                        plt.xlim(0,64)
                        plt.ylim(64,0)

                        plt.savefig(correct_folder + '/' + str(correct) + '.png')

                        plt.gca().clear()

                        correct += 1 

                else:
                    miss_img = inputs[j]

                    img = miss_img.to('cpu').detach().numpy()
                    img = np.transpose(img,(1,2,0))
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    label = labels[j].item()
                    predict = predicted[j].item()

                    ax_pos = plt.axis()  
                    plt.text(ax_pos[0] + 5, ax_pos[2] + 5, 'Label:' + classes[label] + ' Pred:' + classes[predict], color="cyan") 

                    plt.imshow(img * 255)
                    plt.xlim(0,64)
                    plt.ylim(64,0)

                    plt.savefig(miss_folder + '/' + str(miss_num) + '.png')

                    plt.gca().clear()

                    miss_num += 1

    logger.info(f'Total:{total}')
    logger.info(f'Correct:{correct}')
    logger.info(f'Miss:{miss_num}')

    acc = 100 * correct / total
    logger.info(f'Accuracy of test images: {acc:.1f}')
    for i in range(num_class):
            logger.info(f'Accuracy of {classes[i]} : {class_correct[i] / class_total[i]:.1%}')

    ndarray_pred_label = pred_label.to('cpu').detach().numpy().copy()
    ndarray_true_label = true_label.to('cpu').detach().numpy().copy()

    conf_mat = confusion_matrix(ndarray_true_label, ndarray_pred_label)
    logger.info(conf_mat)

    os.makedirs("./results", exist_ok=True)
    f = open('./results/' + file_name+'_Test_Result.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['Test Accuracy'])
    writer.writerow([acc])
    for i in range(num_class):
        writer.writerow(['Test Accuracy of ' + classes[i]])
        writer.writerow([class_correct[i] / class_total[i]*100])
    writer.writerow(['Test Confusion Matrix'])
    for i in range(num_class):
        writer.writerow([conf_mat[i][:]])

    f.close()

    return None


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_name", default='param0', help="setting yaml file name")
    args = parser.parse_args()

    yaml_file = args.yaml_name + ".yaml"
    base_parameters_dir = "./parameters"

    setting_yaml_file = os.path.join(base_parameters_dir, yaml_file)
    parameters = Parameters(setting_yaml_file)

    set_random_seed(parameters.seed)

    file_name = make_file_name(parameters)
    os.makedirs("./logs", exist_ok=True)

    logger.info(file_name)
    logger.info('Loading data ...')

    test(parameters)

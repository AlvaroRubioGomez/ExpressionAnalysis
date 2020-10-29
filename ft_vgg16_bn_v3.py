import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import sys
from torch.utils.tensorboard import SummaryWriter #tensorboard

import argparse

import wandb #weight&bias

#Arguments
def parse_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument("--img_root", type=str, 
                        help="dataset directory", required=True)
    parser.add_argument("--model", type=str, 
                        help="Pretained model selection",
                        choices=[
                            "Resnet18",
                            "vgg_16_bn"], default='Resnet18')    
    parser.add_argument("-v", "--verbose", 
                        help="increase output verbosity", action="store_true") 
    parser.add_argument("--hyperparam", type=int, nargs=2, 
                        help="Couple of integers to define batch size and number of epochs in this order", default=[128,25])
    parser.add_argument("--lr", type=float,
                        help="Learning rate", default=0.00001)
    parser.add_argument("--lr_scheduler", type=float, nargs=3,
                        help="Learning scheduler selected. Three float numbers to define type, step size and gamma in this order.\n Type:(0) None, (1)Step",
                        default=[1,10,0.1])
    parser.add_argument("--opt", type=str, 
                        help="Optimizer selection",
                        choices=[
                            "Adam",
                            "SGD"], default='Adam')

    return parser.parse_args()

    
#Data loader
def get_data(batch_size, img_root, verbose):

    #Define transformation that you wish to apply on image
    data_transforms = transforms.Compose([
                                        #transforms.Resize((224,224)),
                                        transforms.ColorJitter(brightness=0.5),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()
                                        #,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    #Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(root=img_root , transform=data_transforms)

    # Create train and test splits
    # We will create a 80:20 % train:test split
    num_samples = len(image_datasets)
    if(verbose):
        print("number of samples:", num_samples)

    training_samples = int(num_samples * 0.8 + 1)
    test_samples = num_samples - training_samples

    training_data, test_data = torch.utils.data.random_split(image_datasets,[training_samples, test_samples])

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader

#check data
def check_data(data, num_classes):    
    label_list = np.zeros([1,num_classes])   
    for _, labels in data:
        #print(labels)
        for num in labels:
            label_list[0][num.item()] += 1                
    
    return label_list


#Load model
def load_model(num_classes, model_name, verbose):

    if(model_name=='vgg_16_bn'):
        #Load the vgg16 pretrained
        model = models.vgg16_bn(pretrained=True) 

        #Output layer       
        num_ftrs = model.classifier[6].in_features        
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) # modified output

    elif(model_name=='Resnet18'):
        #Load the Resnet18 pretrained
        model = models.resnet18(pretrained=True)   

        #Output layer 
        num_ftrs = model.fc.in_features        
        model.fc = nn.Linear(num_ftrs, num_classes) # modified output
    
    else:
        print("No model selected")
        sys.exit()
    
    if(verbose):
        print("Model:")
        print(model)
        print("----------------------------------------------")

    return model


#Cost function
def get_cost_function():
    cost_function = nn.CrossEntropyLoss()    #SoftMax is already included inside the CrossEntropy method
    return cost_function


#Optimizer
def get_optimizer(model, learning_rate, optimizer_type, verbose):

    if(optimizer_type=='SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    elif(optimizer_type=='Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    else:
        print("No optimizer selected")
        sys.exit()
    
    if(verbose):
        print("optimizer params:", optimizer.state_dict)
    
    return optimizer


#Test (of a batch)
def test(model, test_loader, cost_function, device, verbose):
    n_samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    model.eval() # Strictly needed if network contains layers which has different behaviours between train and test
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):
            # Load data into GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Apply the loss
            loss = cost_function(outputs, labels)

            # Better print something
            n_samples+= images.shape[0]
            cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(labels).sum().item()              

            #test
            del images, labels, outputs, loss
          

    if(verbose):
        print(f'test_loss:{cumulative_loss/n_samples}')
        print(f'test_accuracy:{cumulative_accuracy/n_samples*100}')

    return cumulative_loss/n_samples, cumulative_accuracy/n_samples*100


#Train of a single batch
def batch_train(model, images, labels, optimizer, cost_function, device):   #Train for a single batch 

    cumulative_loss = 0
    n_correct = 0
    n_samples = 0

    model.train()  # Strictly needed if network contains layers which has different behaviours between train and test 
    
    #Load data into GPU     
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = cost_function(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update loss and correct
    cumulative_loss += loss.item()
    _, predicted = torch.max(outputs, 1)     
    n_samples += labels.size(0)   
    n_correct += (predicted == labels).sum().item()        

    return cumulative_loss/n_samples, n_correct/n_samples*100



def get_grid_images(data_loader, writer, tittle, verbose):
    #Tensorboard grid images
    img_batch_obj = iter(data_loader)
    example_data, example_targets = next(img_batch_obj)
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('test loader batch', img_grid)
    writer.close()

    if(verbose):
        print("Image shape:", example_data.size())

def get_confusion_matrix(nb_classes, data_loader, model, device, verbose):

    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    #Test
    if(verbose):
        print("check", check_data(data_loader, nb_classes))

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    if(verbose):
        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1))

        print("check", confusion_matrix.sum(1))



def main(args):  

    #Arguments
    batch_size, num_epochs = args.hyperparam    
    learning_rate = args.lr
    lr_scheduler_flag, step_size, gamma = args.lr_scheduler   
    num_classes = 7
    img_root = args.img_root    

    #weight&bias
    wandb.init(project="facial-expressions-project") 
    wandb.config.batch_size = batch_size 
    wandb.config.num_epochs = num_epochs
    wandb.config.learning_rate = learning_rate
    wandb.config.step_size = step_size
    wandb.config.gamma = gamma
            
    #Data loader
    train_loader, test_loader = get_data(batch_size=batch_size, img_root=img_root, verbose=args.verbose) 
    
    #classes
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    
    #Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Tensorboard configuration
    #comment = f'm:{args.model}, LR:{learning_rate},BS:{batch_size}, e:{num_epochs}, SS:{step_size}, G:{gamma}'
    #writer_train = SummaryWriter('runs/train/' + comment)
    #writer_test = SummaryWriter('runs/test/' + comment)

    #Load model
    model = load_model(num_classes, args.model, args.verbose)

    #Load model into GPU
    model = model.to(device)

    #Optimizer and cost function
    optimizer = get_optimizer(model=model, learning_rate=learning_rate, 
                                optimizer_type=args.opt, verbose = args.verbose)
    cost_function = get_cost_function()

    #scheduler
    if (lr_scheduler_flag != 0):
        step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma) #Every 'step_size' epochs, learning rate is multiply by gamma.

    #Grid images to tensorboard
    #get_grid_images(data_loader=test_loader, writer=writer_test, tittle='test loader batch', verbose = args.verbose)      
        
    #Before training
    print('Before training:')
    train_loss, train_accuracy = test(model, train_loader, cost_function, device, args.verbose)
    test_loss, test_accuracy = test(model, test_loader, cost_function, device, args.verbose)    

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    #delete
    del train_loss, train_accuracy, test_loss, test_accuracy

    #Training
    n_total_steps = len(train_loader)
    step = 0
    for epoch in range(num_epochs):
        #print("epoch:",epoch)
        for batch_id, (images, labels) in enumerate(train_loader):
            #print("batch:", batch_id)
            #Batch train loss and accuracy
            train_loss, train_acc = batch_train(model=model, images=images, labels=labels, 
                            optimizer=optimizer, cost_function=cost_function, device=device)
            #print("train done")           

            if (batch_id+1) % 100 == 0: #print each quarter
                
                 #Batch test loss and accuracy
                test_loss, test_acc = test(model=model, test_loader=test_loader, cost_function=cost_function, device=device, verbose=args.verbose)
                
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_id+1}/{n_total_steps}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc}')

                #Write tensorboard logs               
                #writer_train.add_scalar('Loss', train_loss, global_step=step)
                #writer_train.add_scalar('Accuracy', train_acc, global_step=step)
                #writer_test.add_scalar('Loss', test_loss, global_step=step)
                #writer_test.add_scalar('Accuracy', test_acc, global_step=step)

                #weight&bias
                wandb.log({
                    "Train loss": train_loss,
                    "Train accuracy": train_acc})
                wandb.log({
                    "Test loss": test_loss,
                    "Test accuracy": test_acc})

                #delete
                del test_loss, test_acc
            
            #Increase step
            step += 1

            #delete
            del train_loss, train_acc
        
        #Apply scheduler
        if (lr_scheduler_flag != 0):
            step_lr_scheduler.step()
            if(args.verbose):
                print("LR:", step_lr_scheduler.get_last_lr())

    #Close writer
    #writer_test.close()
    #writer_train.close()


    print('After training:')
    train_loss, train_accuracy = test(model, train_loader, cost_function, device, args.verbose)
    test_loss, test_accuracy = test(model, test_loader, cost_function, device, args.verbose)

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    #Confusion matrix
    get_confusion_matrix(num_classes, test_loader, model, device, args.verbose)



#-------------------------------------------------------------------

if __name__ == '__main__':
    #dataset_path = '/data/agomez/FER2013'
    #dataset_path = 'C:/Users/Alvaro/Desktop/FER2013/FER'
    args = parse_args()
    if(args.verbose):
        print("img_root:", args.img_root) 
        print("model:", args.model)
        print("Batch size:", args.hyperparam[0],"Epochs:", args.hyperparam[1])
        print("Learning rate:", args.lr)
        print("lr_Scheduler:", args.lr_scheduler[0], "Step size:", args.lr_scheduler[1],
                "Gamma:", args.lr_scheduler[2])
        print("optimizer:", args.opt)     
        
    main(args)



    









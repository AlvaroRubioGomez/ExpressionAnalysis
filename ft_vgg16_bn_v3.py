import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import sys
from torch.utils.tensorboard import SummaryWriter

#Data loader
def get_data(batch_size, img_root):

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
    print(num_samples)
    training_samples = int(num_samples * 0.8 + 1)
    test_samples = num_samples - training_samples

    training_data, test_data = torch.utils.data.random_split(image_datasets,[training_samples, test_samples])

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader


#Load model
def load_model(num_classes):

    #Load the vgg16 pretrained
    #vgg_16_bn
    #model = models.vgg16_bn(pretrained=True) 
    #resnet16
    model = models.resnet18(pretrained=True)   

    #Output layer
    #vgg_16_bn
    #num_ftrs = model.classifier[6].in_features        
    #model.classifier[6] = nn.Linear(num_ftrs, num_classes) # modified output
    #resnet16
    num_ftrs = model.fc.in_features        
    model.fc = nn.Linear(num_ftrs, num_classes) # modified output

    return model


#Cost function
def get_cost_function():
    cost_function = nn.CrossEntropyLoss()    #SoftMax is already included inside the CrossEntropy method
    return cost_function


#Optimizer
def get_optimizer(model, learning_rate, optimizer_type='Adam', show=False):

    if(optimizer_type=='SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif(optimizer_type=='Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("No optimizer selected")
        sys.exit()
    
    if(show):
        print("optimmizer:", optimizer.state_dict)
    
    return optimizer


#Test (of a batch)
def test(model, test_loader, cost_function, device, show=False):
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
            #print(batch_id)

            #test
            del images, labels, outputs, loss
          

    if(show):
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



def get_grid_images(data_loader, writer, tittle, show=False):
    #Tensorboard grid images
    img_batch_obj = iter(data_loader)
    example_data, example_targets = next(img_batch_obj)
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('test loader batch', img_grid)
    writer.close()

    if(show):
        print("Image shape:", example_data.size())



def main(batch_size = 128, 
        learning_rate = 0.00001, 
        step_size = 10,
        gamma = 0.1,
        num_epochs = 25,
        num_classes = 7,
        img_root = None):
    
        
    #Data loader
    train_loader, test_loader = get_data(batch_size=batch_size, 
                                       img_root=img_root)
    
    #classes
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    
    #Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Tensorboard configuration
    comment = f'LR:{learning_rate},BS:{batch_size}, SS:{step_size}, G:{gamma}, No normalization, Resnet18, dataaugx3'
    writer_train = SummaryWriter('runs/train/' + comment)
    writer_test = SummaryWriter('runs/test/' + comment)

    #Load model
    model = load_model(num_classes)

    #Load model into GPU
    model = model.to(device)

    #Optimizer and cost function
    optimizer = get_optimizer(model=model, learning_rate=learning_rate, 
                                optimizer_type='Adam', show=True)
    cost_function = get_cost_function()

    #scheduler
    step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma) #Every 'step_size' epochs, learning rate is multiply by gamma.

    #Grid images to tensorboard
    get_grid_images(data_loader=test_loader, writer=writer_test, tittle='test loader batch', show=True)      
        
    #Before training
    print('Before training:')
    train_loss, train_accuracy = test(model, train_loader, cost_function, device)
    test_loss, test_accuracy = test(model, test_loader, cost_function, device)    

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
                test_loss, test_acc = test(model=model, test_loader=test_loader, cost_function=cost_function, device=device)
                #print("test done")

                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_id+1}/{n_total_steps}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc}')

                #Write tensorboard logs               
                writer_train.add_scalar('Loss', train_loss, global_step=step)
                writer_train.add_scalar('Accuracy', train_acc, global_step=step)
                writer_test.add_scalar('Loss', test_loss, global_step=step)
                writer_test.add_scalar('Accuracy', test_acc, global_step=step)

                #delete
                del test_loss, test_acc
            
            #Increase step
            step += 1

            #delete
            del train_loss, train_acc
        
        #Apply scheduler
        step_lr_scheduler.step()
        print("LR:", step_lr_scheduler.get_last_lr())

    #Close writer
    writer_test.close()
    writer_train.close()


    print('After training:')
    train_loss, train_accuracy = test(model, train_loader, cost_function, device)
    test_loss, test_accuracy = test(model, test_loader, cost_function, device)

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')



#-------------------------------------------------------------------

if __name__ == '__main__':
    dataset_path = '/data/agomez/FER2013'
    #dataset_path = 'C:/Users/Alvaro/Desktop/FER2013/FER'
    main(img_root=dataset_path)



    









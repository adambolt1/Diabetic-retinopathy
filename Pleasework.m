

% Two-class Datapath
two_class_datapath = 'Train Dataset Two Classes';


imds=imageDatastore(two_class_datapath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);

% Visualize random 20 images
perm=randperm(num_images,20);
figure;
for idx=1:20
    
    subplot(4,5,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end
% Split the Training and Testing Dataset
train_percent=0.80;
[imdsTrain,imdsTest]=splitEachLabel(imds,train_percent,'randomize');
 
% Split the Training and Validation
valid_percent=0.20;
[imdsValid,imdsTrain]=splitEachLabel(imdsTrain,valid_percent,'randomize');
% Converting images to 299 x 299 to suit the network
augimdsTrain = augmentedImageDatastore([299 299 3],imdsTrain);
augimdsValid = augmentedImageDatastore([299 299 3],imdsValid);

% Set the training options
options = trainingOptions('adam','MaxEpochs',2,'MiniBatchSize',24,...
'Plots','training-progress','Verbose',0,'ExecutionEnvironment','parallel',...
'ValidationData',augimdsValid,'ValidationFrequency',50,'ValidationPatience',3);

% replace lgraph_1 with neural network which has been modified for this
% code from the deep network designer
netTransfer = trainNetwork(augimdsTrain,lgraph_1,options);
% Reshape the test images match with the network 
augimdsTest = augmentedImageDatastore([299 299 3],imdsTest);
%%
 

% Predict Test Labels
[predicted_labels,posterior] = classify(netTransfer,augimdsTest);

% Actual Labels
actual_labels = imdsTest.Labels;

% Confusion Matrix
figure
%replace title with correct title corresponding with network
plotconfusion(actual_labels,predicted_labels)
title('Confusion Matrix: Inception v2-Resnet ');
% Training Data path
datapath='train_images\';

test_labels=double(nominal(imdsTest.Labels));
[fp_rate,tp_rate,T,AUC] = perfcurve(test_labels,posterior(:,2),2);
figure;
plot(fp_rate,tp_rate,'b-');
hold on;
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');
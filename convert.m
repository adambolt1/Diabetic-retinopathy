% Two-class Data path
two_class_datapath='Train Dataset Two Classes\';
myFolder = 'C:\Users\adamy\Desktop\MATLAB\train_images\';
% Class Names
class_names={'No','Yes'};
mkdir(sprintf('%s%s',two_class_datapath,class_names{1}))
mkdir(sprintf('%s%s',two_class_datapath,class_names{2}))

% Read the Excel Sheet with Labels
[num_data,text_data]=xlsread('train.xlsx');

% Determine the Labels
train_labels=num_data(:,2)

% Merge all labels marked into Mild, Medium, Severe and Proliferative DR 
% into a single category 'Yes' 
train_labels(train_labels~=0.0)=2;
% Rest of the dataset belongs to 'No' category
train_labels(train_labels==0.0)=1;

% Filename
filename=text_data(2:end,1);

% Now, write these images 2-folders 'Yes' or 'No' for us to develop a deep
% learning architecture utilizing Deep learning toolbox
% Determine the Files put them in separate folder
for idx=1:length(filename)
    %ended at 2114
  
   
    
    % Read the image
    current_filename=strrep(filename{idx}, char(39), '');
    baseFileName = sprintf('%s.png', current_filename);
    inputFullFileName = fullfile(myFolder, baseFileName);
    fprintf('Reading in image : "%s".\n', inputFullFileName);
    img = imread(inputFullFileName);
     image(img);
     axis('on', 'image');
     impixelinfo;
        drawnow;
       % Write the image in the respective folder
     outputFolder = fullfile(two_class_datapath, class_names{train_labels(idx)});
    outputFullFileName = fullfile(outputFolder, baseFileName);
    fprintf('Saving image to : "%s".\n', outputFullFileName);
    imwrite(img, outputFullFileName);
    
    clear img;
    
end
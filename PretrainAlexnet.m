clear all
clc

imds = imageDatastore('train', 'LabelSource','foldernames','IncludeSubfolders', true);
test = imageDatastore('test', 'LabelSource','foldernames','IncludeSubfolders', true);

augmenter = imageDataAugmenter( ...
'RandXReflection',true, ...
'RandRotation',[-180 180],...
'RandXScale',[1 4], ...
'RandYReflection',true, ...
'RandYScale',[1 4])
imageSize = [112 112 3];
datastore = augmentedImageDatastore(imageSize,imds, ...
'DataAugmentation',augmenter)

layers = [ ... 
imageInputLayer([112 112 3]) 
convolution2dLayer([11 11],96,'Stride',[4 4],'Padding',[0 0 0 0]) 
reluLayer 
crossChannelNormalizationLayer(5) 
maxPooling2dLayer(3,'Stride',[2 2],'Padding',[0 0 0 0]) 
convolution2dLayer([5 5],256,'Stride',[1 1],'Padding',[2 2 2 2]) 
reluLayer 
crossChannelNormalizationLayer(5) 
maxPooling2dLayer(3,'Stride',[2 2],'Padding',[0 0 0 0]) 
convolution2dLayer([3 3],384,'Stride',[1 1],'Padding',[1 1 1 1]) 
reluLayer 
convolution2dLayer([3 3],384,'Stride',[1 1],'Padding',[1 1 1 1]) 
reluLayer 
convolution2dLayer([3 3],256,'Stride',[1 1],'Padding',[1 1 1 1]) 
reluLayer 
maxPooling2dLayer(3,'Stride',[2 2],'Padding',[0 0 0 0]) 
fullyConnectedLayer(4096) 
reluLayer 
dropoutLayer(0.5) 
fullyConnectedLayer(4096) 
reluLayer 
dropoutLayer(0.5) 
fullyConnectedLayer(10) 
softmaxLayer 
classificationLayer];

options = trainingOptions('sgdm', ...
'MaxEpochs',300,...
'InitialLearnRate',1e-4, ...
'Verbose',true, ...
'Plots','training-progress');

net = trainNetwork(imds,layers,options);

YTrain = imds.Labels;
YTest = test.Labels;

YPred = classify(net,test);
plotconfusion(YTest,YPred)

figure
idx = randperm(length(test.Files),25);
for i = 1:25
subplot(5,5,i);
I = readimage(test,idx(i));
I2= imresize(I,[112,112],'nearest');
label = YPred(idx(i));
[Pred,scores] = classify(net,I2);
scores = max(double(scores*100));
imshow(I)
title(join([string(Pred),'' ,scores ,'%']))
end

accuracy = sum(YPred == YTest)/numel(YTest)
save net



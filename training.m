% Training Participants part (this part trains the ensemble of DNNs) of the paper titled:
% "Transfer Learning of an Ensemble of DNNs for SSVEP BCI Spellers without
% User-Specific Training"
%% Preliminaries
% Please download benchmark [2] and/or BETA [3] datasets 
% and add folder that contains downloaded files to the MATLAB path.

% [2] Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
% ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and 
% Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.

% [3] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
% benchmark database toward ssvep-bci application,” Frontiers in
% Neuroscience, vol. 14, p. 627, 2020.
%% Specifications (e.g. number of character) of datasets
subban_no=3; % # of subbands/bandpass filters
dataset='Bench'; % 'Bench' or 'BETA' dataset
signal_length=0.4; % Signal length in second
if strcmp(dataset,'Bench')
    totalparticipants=35; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length     
    total_ch=64; % # of channels used at collection of the dataset  
    max_epochs=1000; % # of epochs for first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
elseif strcmp(dataset,'BETA')
    totalparticipants=70;
    totalblock=4;
    totalcharacter=40;
    sampling_rate=250;
    visual_latency=0.13;
    visual_cue=0.5;
    sample_length=sampling_rate*signal_length; %
    total_ch=64;
    max_epochs=800;
    dropout_second_stage=0.7;
    %else %if you want to use another dataset please specify parameters of the dataset 
    % totalsubject= ... ,
    % totalblock= ... ,
    % ...
end

%% Preprocessing 
total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
% To use all the channels set channels to 1:total_ch=64;
[AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalparticipants,totalblock,totalcharacter,sampling_rate,dataset);
% Dimension of AllData:
% (# of channels, # sample length, #subbands, # of characters, # of blocks, # of subjects)
%% Training of Ensemble of DNNs
sizes=size(AllData);
for test_participant=1:totalparticipants
    
    AllParticipants=1:totalparticipants;
    AllParticipants(test_participant)=[]; % Get indexes of training participants
    
    % Construct the DNN structure from:
    % "A Deep Neural Network for SSVEP-Based Brain-Computer Interfaces"
    layers = [ ...
        imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none')
        convolution2dLayer([1,1],1,'WeightsInitializer','ones') % If you use MATLAB R2018b or previous releases, you need to delete (,'WeightsInitializer','ones') and add (layers(2,1).Weights=ones(1,1,sizes(3))) to the line 81.
        convolution2dLayer([sizes(1),1],120,'WeightsInitializer','narrow-normal') % If you use MATLAB R2018b or previous releases you need to delete (,'WeightsInitializer','narrow-normal') from all convolution2dLayer and fullyConnectedLayer definitions. 
        dropoutLayer(0.1)
        convolution2dLayer([1,2],120,'Stride',[1,2],'WeightsInitializer','narrow-normal')
        dropoutLayer(0.1)
        reluLayer
        convolution2dLayer([1,10],120,'Padding','Same','WeightsInitializer','narrow-normal')
        dropoutLayer(0.95)
        fullyConnectedLayer(totalcharacter,'WeightsInitializer','narrow-normal')
        softmaxLayer
        classificationLayer];      
    
    layers(2, 1).BiasLearnRateFactor=0; % At first layer, sub-bands are combined with 1 cnn layer, 
    % bias term basically adds DC to signal, hence there is no need to use 
    % bias term at first layer. Note: Bias terms are initialized with zeros by default.  
    train=AllData(:,:,:,:,:,AllParticipants); %Getting training data
    train=reshape(train,[sizes(1),sizes(2),sizes(3),totalcharacter*totalblock*length(AllParticipants)]);
    
    train_y=y_AllData(:,:,:,AllParticipants);
    train_y=reshape(train_y,[1,totalcharacter*totalblock*length(AllParticipants)]);    
    train_y=categorical(train_y);
    
    % Global DNN training (First Stage Training)
    options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',max_epochs,...
        'MiniBatchSize',100, ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.001,...
        'ExecutionEnvironment','gpu',...
        'Plots','training-progress');    
    global_DNN = trainNetwork(train,train_y,layers,options);  
    
    %if you want to save global DNN:
    %sv_name=['global_DNN_',dataset,'_',num2str(signal_length),'_',int2str(test_participant),'.mat']; 
    %save(sv_name,'global_DNN'); 
    
    
    % Training of Participant specific DNNs (Second Stage Training)
    participants_DNNs=cell(totalparticipants,1);
    for n=1:totalparticipants-1
        participant=AllParticipants(n);
        layers = [ ...
            imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none')
            convolution2dLayer([1,1],1)
            convolution2dLayer([sizes(1),1],120)
            dropoutLayer(dropout_second_stage)
            convolution2dLayer([1,2],120,'Stride',[1,2])
            dropoutLayer(dropout_second_stage)
            reluLayer
            convolution2dLayer([1,10],120,'Padding','Same')
            dropoutLayer(0.95)
            fullyConnectedLayer(totalcharacter)
            softmaxLayer
            classificationLayer];
        % Transfer the weights that learnt in the first-stage training
        layers(2, 1).Weights = global_DNN.Layers(2, 1).Weights;
        layers(3, 1).Weights = global_DNN.Layers(3, 1).Weights;
        layers(5, 1).Weights = global_DNN.Layers(5, 1).Weights;
        layers(8, 1).Weights = global_DNN.Layers(8, 1).Weights;
        layers(10, 1).Weights = global_DNN.Layers(10, 1).Weights;
        
        layers(2, 1).BiasLearnRateFactor=0;      
        layers(3, 1).Bias = global_DNN.Layers(3, 1).Bias;
        layers(5, 1).Bias = global_DNN.Layers(5, 1).Bias;
        layers(8, 1).Bias = global_DNN.Layers(8, 1).Bias;
        layers(10, 1).Bias = global_DNN.Layers(10, 1).Bias;       
       
        % Getting the participant-specific data
        train=AllData(:,:,:,:,:,participant);
        train=reshape(train,[sizes(1),sizes(2),sizes(3),totalcharacter*totalblock*1]);
       
        train_y=y_AllData(:,:,:,participant);
        train_y=reshape(train_y,[1,totalcharacter*totalblock*1]);   
        
        train_y=categorical(train_y);          
        
        options = trainingOptions('adam',... % Specify training options for second-stage training
            'InitialLearnRate',0.0001,...
            'MaxEpochs',1000,...
            'MiniBatchSize',totalcharacter*(totalblock-1), ...
            'Shuffle','every-epoch',...
            'L2Regularization',0.001,...
            'ExecutionEnvironment','gpu');
        net = trainNetwork(train,train_y,layers,options);
        participants_DNNs{participant}=net;
    end
    sv_name=['participants_DNNs_',dataset,'_',num2str(signal_length),'_',int2str(test_participant),'.mat']; 
    save(sv_name,'participants_DNNs');          
end
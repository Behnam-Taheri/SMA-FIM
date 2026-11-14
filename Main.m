clearvars;
close all;
clc;

%% Settings
projectFolder = fileparts(mfilename('fullpath'));
datasetFile = 'PumsbStarDB.mat';
mst = 14714;           
numItems = 432698;      % Exact frequent-itemset count
N = 16000;              % Number of search agents
T = 1000;               % Maxiter
numRuns = 5;
EnableEarlyStopping = true;

%% SMA options
cfg = struct();
cfg.dataFile = fullfile(projectFolder,datasetFile);
cfg.mst = mst;
cfg.useSeed = false;
cfg.useDimReduction = true;
cfg.useLocalSearch = true;
cfg.useBitPackedFitness = true;
cfg.useDownwardClosure = true;
% Just for test For real run Sholud Be false 
cfg.useStepFunction = false;
% Seed files must match the dataset and minimum support.
cfg.seedFolder = fullfile(projectFolder,'PumsbStar_Random_Seed_Pop');
cfg.seedFilePattern = 'PumsbStar_pop_%02d.mat';

%% Runs
fprintf('SMA | Dataset=%s | MST=%d | Runs=%d\n',datasetFile,mst,numRuns);
percent_list = zeros(1,numRuns);
found_list = zeros(1,numRuns);
for i = 1:numRuns
    [Destination_fitness,bestPositions,Convergence_curve,result] = ...
        SMA(N,T,EnableEarlyStopping,i,cfg);
    found_list(i) = size(result,1);
    percent_list(i) = 100*found_list(i)/numItems;
    fprintf('Run %d/%d | Found=%d | Coverage=%.4f%%\n', ...
        i,numRuns,found_list(i),percent_list(i));
end
mean_percent = mean(percent_list);
std_value = std(percent_list);
fprintf('Mean coverage: %.4f%%\n',mean_percent);
fprintf('Std coverage: %.4f%%\n',std_value);

function [Destination_fitness,bestPositions,Convergence_curve,result] = ...
    SMA(N,Max_iter,EnableEarlyStopping,run,cfg)
InitializationTime = 0;
tic;
%% Initialization
if cfg.useSeed

    filename = fullfile(cfg.seedFolder,sprintf(cfg.seedFilePattern,run));

    savedData = load(filename);

    firstPopulation = savedData.InitialPopulation;
    InitializationTime = savedData.InitializationTime;
    mst = cfg.mst;
    firstPopulation = logical(firstPopulation);
    Data = savedData.DBb;
    [nTransactions,dim] = size(Data);
else
    [firstPopulation,Data,mst] = Initialization(N,cfg.useDimReduction,cfg.dataFile,cfg.mst);
    firstPopulation = logical(firstPopulation);
    Data = logical(Data);
    [nTransactions,dim] = size(Data);
end
    itemSupport = full(sum(Data,1));
    Destination_fitness = inf;
    bestPositions = zeros(1,dim);
    Convergence_curve = zeros(1,Max_iter);
    weight = ones(N,dim);
    columnOffsets = (0:dim-1)*N;

    z = 0.03;
    Patience = 100;
    MinIter = min(200,Max_iter);
    FitnessTolerance = 1e-6;
    stagnationCounter = 0;
    executedIterations = 0;

    MaxSubsetChecks = min(4000,max(120,ceil(N*0.12)));
    MaxNeighborTrials = min(2000,max(48,ceil(N*0.08)));

%% BitPackedFitness
    packedData = [];
    popcountTable = [];
    if cfg.useBitPackedFitness
        nBytes = ceil(nTransactions/8);
        packedData = zeros(nBytes,dim,'uint8');
        for bit = 1:8
            rows = bit:8:nTransactions;
            values = uint8(full(Data(rows,:)));
            packedData(1:numel(rows),:) = bitor( ...
                packedData(1:numel(rows),:),bitshift(values,bit-1));
        end
        popcountTable = zeros(256,1);
        for bit = 1:8
            popcountTable = popcountTable + ...
                double(bitget(uint8((0:255)'),bit));
        end
    end
%% Storage discovered frequent itemsets
    seenItemsets = containers.Map('KeyType','char','ValueType','logical');
    rejectedNeighbors = containers.Map('KeyType','char','ValueType','logical');

    resultCapacity = max(2048,N);
    resultBuffer = false(resultCapacity,dim);
    resultCount = 0;
    subsetCursor = 1;
    subsetRemaining = [];

    fprintf('Enhanced SMA-FIM | N=%d | T=%d | MST=%d | dim=%d\n', ...
        N,Max_iter,mst,dim);
    fprintf('Subset checks=%d | Neighbor trials=%d\n', ...
        MaxSubsetChecks,MaxNeighborTrials);

    % Cache supports from the previous iteration.
    previousPopulation = false(0,dim);
    previousSupport = zeros(0,1);

%% Main loop
    for it = 1:Max_iter

        [uniquePop,~,populationIndex] = unique(firstPopulation,'rows','stable');
        uniqueCount = size(uniquePop,1);
        uniqueSupport = zeros(uniqueCount,1);

        [reused,previousIndex] = ismember(uniquePop,previousPopulation,'rows');
        uniqueSupport(reused) = previousSupport(previousIndex(reused));
        pending = find(~reused).';
        if cfg.useBitPackedFitness
            for k = pending
                uniqueSupport(k) = FitnessPacked( ...
                    uniquePop(k,:),packedData,itemSupport,popcountTable);
            end
        else
            for k = pending
                uniqueSupport(k) = FitnessLogical( ...
                    uniquePop(k,:),Data,itemSupport);
            end
        end
        previousPopulation = uniquePop;
        previousSupport = uniqueSupport;

        patternLengths = sum(uniquePop,2);
        isFrequent = uniqueSupport >= mst & patternLengths > 0;

        uniqueObjective = 1 + max(0,mst-uniqueSupport)/mst;
        uniqueObjective(isFrequent) = ...
            -double(patternLengths(isFrequent)) + ...
            0.01*(uniqueSupport(isFrequent)-mst)/nTransactions;
        uniqueObjective(patternLengths == 0) = 2;

        fit = uniqueObjective(populationIndex).';

        frequentIndexes = find(isFrequent);
        newItemsetsFound = false;

        for j = 1:numel(frequentIndexes)
            k = frequentIndexes(j);
            [resultBuffer,resultCount,resultCapacity,isNew] = ...
                RegisterPattern(uniquePop(k,:),seenItemsets, ...
                resultBuffer,resultCount,resultCapacity);
            if isNew
                newItemsetsFound = true;
            end
        end

        if cfg.useDownwardClosure
            subsetChecks = 0;
            while subsetChecks < MaxSubsetChecks && subsetCursor <= resultCount
                seed = resultBuffer(subsetCursor,:);
                if isempty(subsetRemaining)
                    selected = find(seed);
                    if numel(selected) <= 1
                        subsetCursor = subsetCursor+1;
                        continue;
                    end
                    subsetRemaining = selected(randperm(numel(selected)));
                end
                subset = seed;
                subset(subsetRemaining(end)) = false;
                subsetRemaining(end) = [];
                [resultBuffer,resultCount,resultCapacity,isNew] = ...
                    RegisterPattern(subset,seenItemsets, ...
                    resultBuffer,resultCount,resultCapacity);
                subsetChecks = subsetChecks+1;
                newItemsetsFound = newItemsetsFound || isNew;
                if isempty(subsetRemaining)
                    subsetCursor = subsetCursor+1;
                end
            end
        end
%% Agent-guided local search
        if cfg.useLocalSearch
            frequentAgents = find(isFrequent);
            if ~isempty(frequentAgents)
                frequentAgents = frequentAgents(randperm(numel(frequentAgents)));
            end
            agentCursor = 1;

            for trial = 1:MaxNeighborTrials
                if resultCount > 0 && rand < 0.5
                    seed = resultBuffer(randi(resultCount),:);
                elseif agentCursor <= numel(frequentAgents)
                    agentIndex = frequentAgents(agentCursor);
                    seed = uniquePop(agentIndex,:);
                    agentCursor = agentCursor+1;
                elseif resultCount > 0
                    seed = resultBuffer(randi(resultCount),:);
                else
                    break;
                end

                absent = find(~seed);
                if isempty(absent)
                    continue;
                end

                if rand < 0.5
                    addedItem = absent(randi(numel(absent)));
                else
                    [~,order] = sort(itemSupport(absent),'descend');
                    topCount = min(20,numel(absent));
                    addedItem = absent(order(randi(topCount)));
                end

                neighbor = seed;
                neighbor(addedItem) = true;
                key = ItemsetKey(neighbor);
                if isKey(seenItemsets,key) || isKey(rejectedNeighbors,key)
                    continue;
                end
                if cfg.useBitPackedFitness
                    support = FitnessPacked(neighbor,packedData,itemSupport,popcountTable);
                else
                    support = FitnessLogical(neighbor,Data,itemSupport);
                end

                if support >= mst
                    [resultBuffer,resultCount,resultCapacity,isNew] = ...
                        RegisterPattern(neighbor,seenItemsets, ...
                        resultBuffer,resultCount,resultCapacity);
                    if isNew
                        newItemsetsFound = true;
                    end
                else
                    rejectedNeighbors(key) = true;
                end
            end

            % Limit cache size.
            if rejectedNeighbors.Count > 300000
                rejectedNeighbors = containers.Map( ...
                    'KeyType','char','ValueType','logical');
            end
        end
        % Update weights
        [SmellOrder,SmellIndex] = sort(fit);
        worstFitness = SmellOrder(end);
        bestFitness = SmellOrder(1);
        S = worstFitness-bestFitness+eps;
        logFactor = log10(1+(worstFitness-SmellOrder(:))./S);
        signFactor = ones(N,1);
        signFactor(floor(N/2)+1:end) = -1;
        weight(SmellIndex,:) = 1 + ...
            (signFactor.*logFactor).*rand(N,dim);

        previousBestFitness = Destination_fitness;
        if bestFitness < Destination_fitness
            bestPositions = double(firstPopulation(SmellIndex(1),:));
            Destination_fitness = bestFitness;
        end
        Convergence_curve(it) = Destination_fitness;
        executedIterations = it;

        if EnableEarlyStopping
            fitnessImproved = ...
                (previousBestFitness-Destination_fitness) > FitnessTolerance;
            if fitnessImproved || newItemsetsFound
                stagnationCounter = 0;
            else
                stagnationCounter = stagnationCounter+1;
            end
            closurePending = cfg.useDownwardClosure && subsetCursor <= resultCount;
            if it >= MinIter && stagnationCounter >= Patience && ~closurePending
                fprintf('Early stopping at iteration %d\n',it);
                break;
            end
        end
        if it == Max_iter
            break;
        end

        % Update positions
        a = atanh(1-it/Max_iter);
        b = 1-it/Max_iter;
        for i = 1:N
            if rand < z
                if resultCount > 0 && rand < 0.7
                    seed = resultBuffer(randi(resultCount),:);
                    firstPopulation(i,:) = seed;
                    mutationCount = randi(min(dim,3));
                    mutationIndexes = randperm(dim,mutationCount);
                    firstPopulation(i,mutationIndexes) = ...
                        ~firstPopulation(i,mutationIndexes);
                else
                    firstPopulation(i,:) = false;
                    numberOfItems = randi(min(dim,6));
                    selected = randperm(dim,numberOfItems);
                    firstPopulation(i,selected) = true;
                end
            else
                p = tanh(abs(fit(i)-Destination_fitness));
                vb = -a+2*a*rand(1,dim);
                vc = -b+2*b*rand(1,dim);
                r = rand(1,dim);
                A = randi(N,1,dim);
                B = randi(N,1,dim);
                mask = r < p;
                res = vc.*double(firstPopulation(i,:));

                if any(mask)
                    idxA = A(mask)+columnOffsets(mask);
                    idxB = B(mask)+columnOffsets(mask);
                    res(mask) = bestPositions(mask) + vb(mask).*( ...
                        weight(i,mask).*double(firstPopulation(idxA)) ...
                        -double(firstPopulation(idxB)));
                end
                if cfg.useStepFunction
                    firstPopulation(i,:) = res > 0;
                else
                    prob = 1./(1+exp(-res));
                    firstPopulation(i,:) = rand(1,dim) < prob;
                end
            end
        end

        if mod(it,100) == 0
            fprintf('Iter %d/%d | Frequent itemsets=%d | Time=%.1fs\n', ...
                it,Max_iter,resultCount,toc);
        end
    end

    Convergence_curve = Convergence_curve(1:executedIterations);
    result = resultBuffer(1:resultCount,:);
    fprintf('\nEarly Stopping Enabled: %d\n',EnableEarlyStopping);
    fprintf('Executed iterations: %d\n',executedIterations);
    fprintf('Unique frequent itemsets: %d\n',resultCount);
    fprintf('Runtime: %.4f seconds\n',toc + InitializationTime);

end

    function support = FitnessPacked(x,packedData,itemSupport,popcountTable)
        indexes = find(x);
        if isempty(indexes), support = 0; return; end
        if numel(indexes)==1
            support = itemSupport(indexes);
            return;
        end
        [~,ord] = sort(itemSupport(indexes),'ascend');
        indexes = indexes(ord);
        bits = packedData(:,indexes(1));
        for j=2:numel(indexes)
            bits = bitand(bits,packedData(:,indexes(j)));
            if ~any(bits), support = 0; return; end
        end
        support = sum(popcountTable(double(bits)+1));
    end

    function support = FitnessLogical(x,Data,itemSupport)
        indexes = find(x);
        if isempty(indexes), support=0; return; end
        if numel(indexes)==1, support=itemSupport(indexes); return; end
        support = full(sum(all(Data(:,indexes),2)));
    end
    function [buffer,count,capacity,isNew] = ...
            RegisterPattern(pattern,seen,buffer,count,capacity)
        isNew = false;
        if ~any(pattern)
            return;
        end
        key = ItemsetKey(pattern);
        if isKey(seen,key)
            return;
        end
        seen(key) = true;
        count = count+1;
        if count > capacity
            capacity = capacity*2;
            buffer(capacity,size(buffer,2)) = false;
        end
        buffer(count,:) = pattern;
        isNew = true;
    end

    function key = ItemsetKey(x)
        padded = false(8,ceil(numel(x)/8));
        padded(1:numel(x)) = x;
        bytes = [1 2 4 8 16 32 64 128]*double(padded);
        digits = '0123456789ABCDEF';
        hex = digits([floor(bytes/16); mod(bytes,16)]+1);
        key = reshape(hex,1,[]);
    end

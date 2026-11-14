function [firstPopulation,DBb,mst] = Initialization(N,EnableDimReduction,dataFile,mst)

validateattributes(mst,{'numeric'},{'scalar','integer','positive','finite'});
t = load(dataFile,'DB');
DB = t.DB;
nTransactions = numel(DB);

transactionLengths = cellfun(@numel,DB);
transactionLengths = transactionLengths(:);
allItems = [DB{:}];
[items,~,itemColumns] = unique(allItems(:));
nItems = numel(items);
rowIndices = repelem((1:nTransactions)',transactionLengths);

binarySparse = sparse(rowIndices,itemColumns,1,nTransactions,nItems);
binarySparse = spones(binarySparse);
countTemp = full(sum(binarySparse,1));
if EnableDimReduction
    frequentMask = countTemp >= mst;
else
    frequentMask = true(1,nItems);
end
dim = nnz(frequentMask);
if dim == 0
    error('No frequent items found.');
end
DBb = logical(binarySparse(:,frequentMask));

global DBbinary
DBbinary = binarySparse;

firstPopulation = false(N,dim);
if N <= dim
    selected = randperm(dim,N);
    for i = 1:N
        firstPopulation(i,selected(i)) = true;
    end
else
    firstPopulation(1:dim,:) = logical(eye(dim));
    if dim <= 52 && N > 2^dim-1
        error('Not enough unique itemsets for N agents.');
    end
    seen = containers.Map('KeyType','char','ValueType','logical');
    for k = 1:dim
        seen(InitialPatternKey(firstPopulation(k,:))) = true;
    end
    populationCount = dim;
    failedAttempts = 0;
    maxAttempts = max(1000,20*N);
    while populationCount < N
        r1 = randi(populationCount);
        r2 = randi(populationCount-1);
        if r2 >= r1
            r2 = r2+1;
        end
        %candidate = firstPopulation(r1,:) | firstPopulation(r2,:);
        candidate = xor(firstPopulation(r1,:),firstPopulation(r2,:));
        key = InitialPatternKey(candidate);
        if ~isKey(seen,key)
            seen(key) = true;
            populationCount = populationCount+1;
            firstPopulation(populationCount,:) = candidate;
            failedAttempts = 0;
        else
            failedAttempts = failedAttempts+1;
        end
        if failedAttempts >= maxAttempts
            error('Unable to generate unique agents.');
        end
    end
end
fprintf('Transactions: %d | Original items: %d | Frequent items: %d\n', ...
    nTransactions,nItems,dim);
end

function key = InitialPatternKey(x)
padded = false(8,ceil(numel(x)/8));
padded(1:numel(x)) = x;
bytes = [1 2 4 8 16 32 64 128]*double(padded);
digits = '0123456789ABCDEF';
hex = digits([floor(bytes/16); mod(bytes,16)]+1);
key = reshape(hex,1,[]);
end

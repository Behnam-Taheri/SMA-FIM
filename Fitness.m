function repeat = Fitness(x, FirstDb)

    indexes = find(x == 1);

    if isempty(indexes)
        repeat = 0;
        return;
    end

    repeat = sum(all(FirstDb(:,indexes),2));

end
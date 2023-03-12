function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

for i = 1:size(Lclass)
    cM(Lclass(i),Ltrue(i)) = cM(Lclass(i),Ltrue(i)) + 1;
end

end


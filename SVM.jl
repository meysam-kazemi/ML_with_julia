
# SVM struct
mutable struct SvmStruct
    epoch :: Int;
    lr :: Float32; # Learning Rate
    w :: Vector{Any}; # weigtht
    output :: Vector{Any}; # save outputs
end

function SVM(x,y,lr=1,epoch=100000)
    # Build a SVM model
    svm = SvmStruct(epoch,lr,zeros(size(x,2)),[])
    # TRAIN
    for e in 1:epoch
        for (i,row) in enumerate(eachrow(x))
            val = row' * svm.w
            # Update w
            if y[i] * val < 1
                svm.w = svm.w .+ lr * ((y[i] * row) .- (2*(1/epoch)*svm.w))
            else
                svm.w = svm.w .+ lr * (-2*(1/epoch)*svm.w)
            end
        end
    end
    for row in eachrow(x)
        append!(svm.output,row' * svm.w) # Add the output of the model in the output variable
    end
    return svm
end




# TEST
#Input data
x = Float32.([0 2 -1;-2 4 -1;4 1 -1;1 6 -1;2 4 -1;6 2 -1]);
# Output label
y = [-1 -1 -1 1 1 1];

svm = SVM(x,y)
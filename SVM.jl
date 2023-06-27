
# SVM struct
mutable struct SvmStruct
    epoch :: Int;
    lr :: Float32; # Learning Rate
    w :: Vector{Any}; # weigtht
    train_output :: Vector{Any}; # save outputs
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
        append!(svm.train_output,row' * svm.w) # Add the output of the model in the train_output variable
    end
    return svm
end

function predict(svm::SvmStruct,x)
    output = []
    for row in eachrow(x)
        append!(output,row' * svm.w)
    end
    return sign.(output)
end


# # TEST
# #Input data
# x = Float32.([0 2 -1;-2 4 -1;4 1 -1;1 6 -1;2 4 -1;6 2 -1]);
# # Output label
# y = [-1 -1 -1 1 1 1];
# svm = SVM(x,y)

# testX = Float32.([5 4 -1;-2 -5 -1;3 1 -1;1 1 -1])
# predict(svm,testX)
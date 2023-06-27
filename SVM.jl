
mutable struct SVM
    epoch :: Int
    lr :: Float32
    w :: Vector
    output = []
end

function SVM(x,y,lr=1,epoch=100000)
    SVM(epoch,lr,zeros(size(x,2)))
    for e in 1:epoch
        for (i,row) in enumerate(eachrow(x))
            val = row' * SVM.w
            if y[i] * val < 1
                w = w .+ lr * ((y[i] * row) .- (2*(1/epoch)*SVM.w))
            else
                w = w .+ lr * (-2*(1/epoch)*SVM.w)
            end
        end
    end
    for (i,row) in enumerate(eachrow(x))
        append!(SVM.output,row' * SVM.w)
    end
    return SVM.w,SVM.output
end

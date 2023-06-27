


function SVM(x,y,lr=1,epoch=100000)
    w = zeros(size(x,2))
    out = []
    for e in 1:epoch
        for (i,row) in enumerate(eachrow(x))
            val = row' * w
            if y[i] * val < 1
                w = w .+ lr * ((y[i] * row) .- (2*(1/epoch)*w))
            else
                w = w .+ lr * (-2*(1/epoch)*w)
            end
        end
    end
    for (i,row) in enumerate(eachrow(x))
        append!(out,row' * w)
    end
    return w,out
end

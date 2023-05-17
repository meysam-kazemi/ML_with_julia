import Statistics

function seprate_classes(x,y) # Separate the dataset into a subset of data for each class
    seprated_classes = Dict() # save seprated_classes in a dictionary var
    for i in 1:size(x,1)
        feature = x[i,:,:]' # size(x[i,:])->(4,)  -  size(x[i,:,:])->(4,1)  - size(x[i,:,:])->(1,4)
        class_name = y[i]
        # Check seprated_classes is in class_name or not
        # if not : add the name of this class to seprate_classes(with empty array)
        if class_name ∉ keys(seprated_classes) 
            seprated_classes[class_name] = Array{Float64}(undef,0,4) # empty array : size -> 0,4
        end
        seprated_classes[class_name] = [seprated_classes[class_name];feature];
    end    
    return seprated_classes
end


function std_mean(x) # Calculates standard deviation and mean of features.
    std = []
    mean = []
    for feature in eachcol(x)
        append!(std,Statistics.std(feature))
        append!(mean,Statistics.mean(feature))
    end
    return std , mean
end


function distribution(x,std,mean) # Gaussian Distribution Function
    exponent = exp.(-((x .- mean).^2 ./ (2 .* std .^ 2)))
    return exponent ./ (sqrt(2*π)*std)
end
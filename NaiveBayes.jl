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


function fit(x,y)
    classes = seprate_classes(x,y)
    class_summary = Dict() # Save std and mean
    for item in classes
        class_name , feature_val = item.first , item.second
        std,mean = std_mean(feature_val);
        summary = Dict("mean"=>mean,"std"=>std);
        class_summary[class_name] = Dict("prior_proba"=>length(feature_val)/size(x,1),
                                        "summary"=>summary)
    end
    return class_summary        
end


# Predict classes
function predict(x,class_summary)
    MAPs = []
    for row in eachrow(x)
        joint_proba = Dict()
        for item in class_summary
            class_name , features = item.first,item.second
            total_features = length(features["summary"])
            likelihood =1

            for idx in 1:total_features
                feature = row[idx]
                mean_ = features["summary"]["mean"][idx]
                std_ = features["summary"]["std"][idx]
                normal_proba = distribution(feature,std_,mean_)
                likelihood *= normal_proba
            end
            prior_proba = features["prior_proba"]
            joint_proba[class_name] = prior_proba * likelihood
        end
        MAP = reduce((x,y)->joint_proba[x]>joint_proba[y] ? x : y,keys(joint_proba)) 
        append!(MAPs,MAP)
    end
    return MAPs
end
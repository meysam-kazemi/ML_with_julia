import Plots , Statistics

# Build a simple regression
# The model needs to a loss function
MSE(y,ŷ) = sum((y .- ŷ).^2) / length(y)

function simple_regression(X , Y , epochs , lr = 0.01)
    error = []
    len = length(Y)
    a0 , a1 = (0,0) # Coefficients
    for e in 1:epochs # Training loop
        etoepochs = e / epochs
        print("[$("=" ^ Int(round(40 * etoepochs)) * " " ^ Int(round(40 * (1-etoepochs))))] -> $(round((e*100/epochs),digits=2)) \r")
        Ŷ = a0 .+ a1 .* X # Predict Y
        # Update coefficients
        a0 = a0 -  (lr * 2 * Statistics.mean((Ŷ - Y)))
        a1 = a1 -  (lr * 2 * Statistics.mean((Ŷ - Y) .* X))
        append!(error,MSE(Y,Ŷ))
    end
    # println("Done ☑ ")
    return a0 ,a1 , error
end


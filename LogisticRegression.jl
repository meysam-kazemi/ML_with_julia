import Statistics

sigmoid(a) = @. 1 / (1 + exp(-a)) # Activation Function

 # Loss function(logitcrossentropy)
loss(y,ŷ) = Statistics.mean(-y .* log.(ŷ) .- (1 .- y) .* log.(1 .- ŷ))


function predict(X , theta)
    ŷ = sigmoid(X * theta)
    for i in 1:length(ŷ)
        if ŷ[i] <= 0.5
            ŷ[i] = 0
        else
            ŷ[i] = 1
        end
    end
    return ŷ
end


function LogisticRegression(x,y,epochs,lr)
    losses = zeros(epochs) # Save losses of each epoch

    θ = zeros(size(x,2)) # weights initialization
    for e in 1:epochs
        ŷ = sigmoid(x * θ)
        gradient = (transpose(X) * (ŷ .- y)) ./ length(y) # calculate gradient
        θ .-= lr * gradient # Update theta
        ŷ = (θ .* x) |> sigmoid 
        losses[e] = loss(y,ŷ) # Save loss
    end
    println("loss = $(losses[epochs])")
    return θ , losses
end

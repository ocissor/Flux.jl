"""
    mae(ŷ, y; agg=mean)

Return the loss corresponding to mean absolute error: 

    agg(abs.(ŷ .- y))
"""
mae(ŷ, y; agg=mean) = agg(abs.(ŷ .- y))

"""
    mse(ŷ, y; agg=mean)

Return the loss corresponding to mean square error: 
    
    agg((ŷ .- y).^2)
"""
mse(ŷ, y; agg=mean) = agg((ŷ .- y).^2)

"""
    msle(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))

The loss corresponding to mean squared logarithmic errors, calculated as

    agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)

The `ϵ` term provides numerical stability.
Penalizes an under-predicted estimate more than an over-predicted estimate.
"""
msle(ŷ, y; agg=mean, ϵ=epseltype(ŷ)) = agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)

"""
    huber_loss(ŷ, y; δ=1, agg=mean)

Return the mean of the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
given the prediction `ŷ` and true values `y`.

                 | 0.5 * |ŷ - y|,            for |ŷ - y| <= δ
    Huber loss = |
                 |  δ * (|ŷ - y| - 0.5 * δ), otherwise
"""
function huber_loss(ŷ, y; agg=mean, δ=ofeltype(ŷ, 1))
   abs_error = abs.(ŷ .- y)
   temp = abs_error .<  δ
   x = ofeltype(ŷ, 0.5)
   agg(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp))
end

wsum(w::Nothing, x; dims) = sum(x, dims=dims)
wsum(w::Number, x; dims) = w .* sum(x, dims=dims)
wsum(w::AbstractArray, x; dims) = sum( w .* x, dims=dims)

"""
    crossentropy(ŷ, y; weight=nothing, dims=1, ϵ=eps(eltype(ŷ)), agg=mean)

Return the cross entropy between the given probability distributions;
calculated as

    agg(.-sum(weight .* y .* log.(ŷ .+ ϵ); dims=dims))agg=mean, 

`weight` can be `nothing`, a number or an array.
`weight=nothing` acts like `weight=1` but is faster.

See also: [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)
"""
function crossentropy(ŷ, y; dims=1, agg=mean, ϵ=epseltype(ŷ), weight=nothing)
  agg(.-wsum(weight, y .* log.(ŷ .+ ϵ); dims=dims))
end

"""
    logitcrossentropy(ŷ, y; weight=nothing, agg=mean, dims=1)

Return the crossentropy computed after a [`Flux.logsoftmax`](@ref) operation;
calculated as

    agg(.-sum(weight .* y .* logsoftmax(ŷ; dims=dims); dims=dims))

`logitcrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.crossentropy(softmax(log.(ŷ)), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)
"""
function logitcrossentropy(ŷ, y; dims=1, agg=mean, weight=nothing)
  agg(.-wsum(weight, y .* logsoftmax(ŷ; dims=dims); dims=dims))
end

"""
    binarycrossentropy(ŷ, y; agg=mean, ϵ=epseltype(ŷ))

Return ``-y*\\log(ŷ + ϵ) - (1-y)*\\log(1-ŷ + ϵ)``. The `ϵ` term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)
"""
function binarycrossentropy(ŷ, y; agg=mean, ϵ=epseltype(ŷ))
  agg(@.(-y*log(ŷ+ϵ) - (1-y)*log(1-ŷ+ϵ)))
end

"""
    logitbinarycrossentropy(ŷ, y; agg=mean)

`logitbinarycrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.binarycrossentropy(σ(log(ŷ)), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref)
"""
function logitbinarycrossentropy(ŷ, y; agg=mean)
  agg(@.((1-y)*ŷ - logsigmoid(ŷ)))
end

"""
    kldivergence(ŷ, y; dims=1, agg=mean, ϵ=eps(eltype(ŷ)))

Return the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given arrays interpreted as probability distributions.

KL divergence is a measure of how much one probability distribution is different
from the other.
It is always non-negative and zero only when both the distributions are equal
everywhere.
"""
function kldivergence(ŷ, y; dims=1, agg=mean, ϵ=epseltype(ŷ))
  entropy = agg(sum(y .* log.(y .+ ϵ), dims=dims))
  cross_entropy = crossentropy(ŷ, y; dims=dims, agg=agg, ϵ=ϵ)
  return entropy + cross_entropy
end

"""
    poisson_loss(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ))))

Loss function derived from likelihood for a Poisson random variable with mean
`ŷ` to take value `y`. It is given by

    agg(ŷ .- y .* log.(ŷ .+ ϵ))

[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson_loss(ŷ, y; agg=mean, ϵ=epseltype(ŷ)) = agg(ŷ .- y .* log.(ŷ .+ ϵ))

"""
    hinge(ŷ, y; agg=mean)

Return the [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as

    agg(max.(0, 1 .- ŷ .* y))

See also: [`squared_hinge`](@ref)
"""
hinge(ŷ, y; agg=mean) = agg(max.(0, 1 .-  ŷ .* y))

"""
    squared_hinge(ŷ, y; agg=mean)

Return the squared hinge loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as
    
    agg(max.(0, 1 .- ŷ .* y).^2)

See also: [`hinge`](@ref)
"""
squared_hinge(ŷ, y; agg=mean) = agg(max.(0, 1 .- ŷ .* y).^2)

"""
    dice_coeff_loss(ŷ, y; smooth=1, dims=size(ŷ)[1:end-1], agg=mean)

Return a loss based on the Dice coefficient.
Used in the [V-Net](https://arxiv.org/pdf/1606.04797v1.pdf) architecture 
for image segmentation. 
Current implementation only works for the binary segmentation case.
    
The arrays `ŷ` and `y` contain the predicted and true probabilities respectively
for the foreground to be present in a certain pixel. 
The loss is computed as

    1 - (2*sum(ŷ .* y; dims) .+ smooth) ./ (sum(ŷ.^2 .+ y.^2; dims) .+ smooth)

and then aggregated with `agg` over the batch.
"""
function dice_coeff_loss(ŷ, y; smooth=ofeltype(ŷ, 1), 
                              dims=size(ŷ)[1:end-1],
                              agg=mean)
  f = x -> sum(x, dims=dims)
  agg(1 .- (2 .* f(y .* ŷ) .+ smooth) ./ (f(y.^2 + ŷ.^2) .+ smooth))
end

"""
    tversky_loss(ŷ, y; β=0.7, α=1-β, dims=size(ŷ)[1:end-1] agg=mean)

Return the [Tversky loss](https://arxiv.org/pdf/1706.05721.pdf)
for binary classification. 
The arrays `ŷ` and `y` contain the predicted and true probabilities respectively.
Used with imbalanced data to give more weight to false negatives.
Larger `β` weigh recall higher than precision (by placing more emphasis on false negatives)
Calculated as:

    num = sum(y .* ŷ, dims=dims)
    den = sum(@.(ŷ*y + α*ŷ*(1-y) + β*(1-ŷ)*y)), dims=dims)
    tversky_loss = 1 - num/den

and then aggregated with `agg` over the batch.

When `α+β=1`, it is equal to `1-F_β`, where `F_β` is an F-score.
"""
function tversky_loss(ŷ, y; β=ofeltype(ŷ, 0.7), α=1-β, dims=size(ŷ)[1:end-1], agg=mean)
  f = x -> sum(x, dims=dims)
  agg(1 .- f(ŷ .* y) ./ f(@.(ŷ*y + α*ŷ*(1-y) + β*(1-ŷ)*y))) 
end


# TODO normalise over last dimension is typically what you want to do. 
# Possible deprecation path: `normalise(x; dims=1)` -> `normalise(x; dims)` -> `normalise(x; dims=size(x)[end])`  
"""
    normalise(x; dims=1)

Normalise `x` to mean 0 and standard deviation 1 across the dimensions given by `dims`.
Defaults to normalising over columns.

```jldoctest
julia> a = reshape(collect(1:9), 3, 3)
3×3 Array{Int64,2}:
 1  4  7
 2  5  8
 3  6  9

julia> Flux.normalise(a)
3×3 Array{Float64,2}:
 -1.22474  -1.22474  -1.22474
  0.0       0.0       0.0
  1.22474   1.22474   1.22474

julia> Flux.normalise(a, dims=2)
3×3 Array{Float64,2}:
 -1.22474  0.0  1.22474
 -1.22474  0.0  1.22474
 -1.22474  0.0  1.22474
```
"""
function normalise(x::AbstractArray; dims=1, ϵ=ofeltype(x, 1e-6))
  μ′ = mean(x, dims=dims)
#   σ′ = std(x, dims=dims, mean=μ′, corrected=false) # use this when #478 gets merged
  σ′ = std(x, dims=dims, corrected=false)
  return (x .- μ′) ./ (σ′.+ ϵ)
end

"""
    flatten(x::AbstractArray)

Reshape arbitrarly-shaped input into a matrix-shaped output
preserving the last dimension size. 
Equivalent to `reshape(x, :, size(x)[end])`.
"""
flatten(x::AbstractArray) = reshape(x, :, size(x)[end])

module GeostatInversion

import RandMatFact
import IterativeSolvers
import FDDerivatives
import Optim
import RobustPmap

include("direct.jl")
include("lowrank.jl")
include("lsqr.jl")
include("lm.jl")

function randsvdwithseed(Q, numxis, p, q, seed::Void)
	return RandMatFact.randsvd(Q, numxis, p, q)
end

function randsvdwithseed(Q, numxis, p, q, seed::Int)
	srand(seed)
	return RandMatFact.randsvd(Q, numxis, p, q)
end

function getxis(::Type{Val{:iwantfields}}, samplefield::Function, numfields::Int, numxis::Int, p::Int, q::Int=3, seed=nothing)
	fields = RobustPmap.rpmap(i->samplefield(), 1:numfields; t=Array{Float64, 1})
	lrcm = LowRankCovMatrix(fields)
	Z = randsvdwithseed(lrcm, numxis, p, q, seed)
	xis = Array(Array{Float64, 1}, numxis)
	for i = 1:numxis
		xis[i] = Z[:, i]
	end
	return xis, fields
end

function getxis(samplefield::Function, numfields::Int, numxis::Int, p::Int, q::Int=3, seed=nothing)
	xis, _ = getxis(Val{:iwantfields}, samplefield, numfields, numxis, p, q, seed)
	return xis
end

function getxis(Q::Matrix, numxis::Int, p::Int, q::Int=3, seed=nothing)#numxis is the number of xis, p is oversampling for randsvd accuracy, q is the number of power iterations -- see review paper by Halko et al
	xis = Array(Array{Float64, 1}, numxis)
	Z = randsvdwithseed(Q, numxis, p, q, seed)
	for i = 1:numxis
		xis[i] = Z[:, i]
	end
	return xis
end

function srga(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector, Kred; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6, pcgafunc=pcgadirect, callback=(s, obs_cal)->nothing)
	S = sprandn(Kred, length(y), ceil(Int, log(length(y))) / Kred)
	scale!(S, 1 / sqrt(length(y)))
	return pcgafunc(x->S * forwardmodel(x), s0, X, xis, S * R * S', S * y; maxiters=maxiters, delta=delta, xtol=xtol, callback=callback)
end

function rga(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector, S; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6, pcgafunc=pcgadirect, callback=(s, obs_cal)->nothing)
	return pcgafunc(x->S * forwardmodel(x), s0, X, xis, S * R * S', S * y; maxiters=maxiters, delta=delta, xtol=xtol, callback=callback)
end

pcga = pcgadirect
#TODO implement a pcga that adaptively selects between lsqr and direct based on the number of observations

end

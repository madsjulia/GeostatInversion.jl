module GeostatInversion

import RandMatFact
import IterativeSolvers
import FDDerivatives
import Optim

include("direct.jl")
include("lowrank.jl")
include("lsqr.jl")
include("lm.jl")

function getxis(samplefield::Function, numfields::Int, numxis::Int, p::Int, q::Int=3)
	fieldsstupidanytype = pmap(i->samplefield(), zeros(numfields))
	fields = Array(Array{Float64, 1}, numfields)
	for i = 1:numfields
		fields[i] = fieldsstupidanytype[i]
	end
	lrcm = LowRankCovMatrix(fields)
	Z = RandMatFact.randsvd(lrcm, numxis, p, q)
	xis = Array(Array{Float64, 1}, numxis)
	for i = 1:numxis
		xis[i] = Z[:, i]
	end
	return xis
end

function getxis(Q::Matrix, numxis::Int, p::Int, q::Int=3)#numxis is the number of xis, p is oversampling for randsvd accuracy, q is the number of power iterations -- see review paper by Halko et al
	xis = Array(Array{Float64, 1}, numxis)
	Z = RandMatFact.randsvd(Q, numxis, p, q)
	for i = 1:numxis
		xis[i] = Z[:, i]
	end
	return xis
end

function rga(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector, S; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6, pcgafunc=pcgadirect)
	return pcgafunc(x->S * forwardmodel(x), s0, X, xis, S * R * S', S * y; maxiters=maxiters, delta=delta, xtol=xtol)
end

pcga = pcgadirect
#TODO implement a pcga that adaptively selects between lsqr and direct based on the number of observations

end

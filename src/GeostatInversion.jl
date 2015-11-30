module GeostatInversion

import RMF
import IterativeSolvers

include("lowrank.jl")

function getxis(samplefield::Function, numfields::Int, numxis::Int, p::Int, q::Int=3)
	fields = Array(Array{Float64, 1}, numfields)
	for i = 1:numfields
		fields[i] = samplefield()
	end
	lrcm = LowRankCovMatrix(fields)
	Z = RMF.randsvd(lrcm, numxis, p, q)
	xis = Array(Array{Float64, 1}, numxis)
	for i = 1:numxis
		xis[i] = Z[:, i]
	end
	return xis
end

function getxis(Q::Matrix, numxis::Int, p::Int, q::Int=3)#numxis is the number of xis, p is oversampling for randsvd accuracy, q is the number of power iterations -- see review paper by Halko et al
	xis = Array(Array{Float64, 1}, numxis)
	Z = RMF.randsvd(Q, numxis, p, q)
	for i = 1:numxis
		xis[i] = Z[:, i]
	end
	return xis
end

function rga(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector, S; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6, pcgafunc=pcgadirect)
	return pcgafunc(x->S * forwardmodel(x), s0, X, xis, S * R * S', S * y; maxiters=maxiters, delta=delta, xtol=xtol)
end

#Inputs: 
#forwardmodel - param to obs map h(s)
#s0 - initial guess
#X - mean of parameter prior (replace with B*X drift matrix later for p>1)
#xis - K columns of Z = randSVDzetas(Q,K,p,q) where Q approx= ZZ^T
#R - covariance of measurement error (data misfit term)
#y - data vector
#Optional Args
#maxIter - maximum # of PCGA iterations
#delta - the finite difference step size
function pcgadirect(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6)
	HQH = Array(Float64, length(y), length(y))
	converged = false
	s = s0
	itercount = 0
	while !converged && itercount < maxiters
		olds = s
		s = pcgadirectiteration!(HQH, forwardmodel, s, X, xis, R, y, delta)
		if norm(s - olds) < xtol
			converged = true
		end
		itercount += 1
	end
	return s
end

function pcgadirectiteration!(HQH::Matrix, forwardmodel::Function, s::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector, delta::Float64)
	p = 1#p = 1 because X is a vector rather than a full matrix
	paramstorun = Array(Array{Float64, 1}, length(xis) + 3)
	for i = 1:length(xis)
		paramstorun[i] = s + delta * xis[i]
	end
	paramstorun[length(xis) + 1] = s + delta * X
	paramstorun[length(xis) + 2] = s + delta * s
	paramstorun[length(xis) + 3] = s
	results = pmap(forwardmodel, paramstorun) 
	hs = results[length(xis) + 3]
	fill!(HQH, 0.)
	for i = 1:length(xis)
		etai = (results[i] - hs) / delta
		BLAS.ger!(1., etai, etai, HQH)
	end
	HX = (results[length(xis)+1] - hs) / delta
	Hs = (results[length(xis)+2] - hs) / delta
	b = [y - hs + Hs; zeros(p)];
	bigA = [(HQH + R) HX; transpose(HX) zeros(p, p)]
	x = pinv(bigA) * b
	beta_bar = x[end]
	xi_bar = x[1:end-1]
	s = X * beta_bar
	for i = 1:length(xis)#add HQ' * xi_bar to s
		etai = (results[i] - hs) / delta
		s += xis[i] * dot(etai, xi_bar)
	end
	return s
end

function pcgalsqr(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6)
	converged = false
	s = s0
	itercount = 0
	while !converged && itercount < maxiters
		olds = s
		s = pcgalsqriteration(forwardmodel, s, X, xis, R, y, delta)
		if norm(s - olds) < xtol
			converged = true
		end
		itercount += 1
	end
	return s
end

function pcgalsqriteration(forwardmodel::Function, s::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R, y::Vector, delta::Float64)
	p = 1#p = 1 because X is a vector rather than a full matrix
	paramstorun = Array(Array{Float64, 1}, length(xis) + 3)
	for i = 1:length(xis)
		paramstorun[i] = s + delta * xis[i]
	end
	paramstorun[length(xis) + 1] = s + delta * X
	paramstorun[length(xis) + 2] = s + delta * s
	paramstorun[length(xis) + 3] = s
	results = pmap(forwardmodel, paramstorun)
	hs = results[length(xis) + 3]
	etas = Array(Array{Float64, 1}, length(xis))
	for i = 1:length(xis)
		etas[i] = (results[i] - hs) / delta
	end
	HX = (results[length(xis)+1] - hs) / delta
	Hs = (results[length(xis)+2] - hs) / delta
	b = [y - hs + Hs; zeros(p)];
	bigA = PCGALowRankMatrix(etas, HX, R)
	x = IterativeSolvers.lsqr(bigA, b)[1]
	beta_bar = x[end]
	xi_bar = x[1:end-1]
	s = X * beta_bar
	for i = 1:length(xis)#add HQ' * xi_bar to s
		etai = (results[i] - hs) / delta
		s += xis[i] * dot(etai, xi_bar)
	end
	return s
end

pcga = pcgadirect
#TODO implement a pcga that adaptively selects between lsqr and direct based on the number of observations

end

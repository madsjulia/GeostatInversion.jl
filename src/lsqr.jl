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

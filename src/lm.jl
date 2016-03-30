function getmodelparams(x, X, xis::Array{Array{Float64, 1}})
	modelparams = X * x[end]
	for i = 1:length(xis)
		BLAS.axpy!(x[i], xis[i], modelparams)
	end
	return modelparams
end

function pcgalm(forwardmodel::Function, s0::Vector, X::Vector, numxis::Int, getmodelparamsshort::Function, Rdiag::Vector, y::Vector; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6)
	function lm_f(x::Vector)
		modelparams = getmodelparamsshort(x, X)
		modelpredictions = forwardmodel(modelparams)
		return sqrt(Rdiag) .* (modelpredictions - y)
	end
	lm_g = FDDerivatives.makejacobian(lm_f)
	opt = Optim.levenberg_marquardt(lm_f, lm_g, [zeros(numxis); 1.])
	result = getmodelparamsshort(opt.minimum, X)
	return result
end

function pcgalm(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, Rdiag::Vector, y::Vector; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6)
	masternode = myid()
	global ___global___geostatinversion___xis = xis
	function getmodelparamsshort(x, X)
		xref = @spawnat myid() x
		Xref = @spawnat myid() X
		return fetch(@spawnat masternode GeostatInversion.getmodelparams(fetch(xref), fetch(Xref), ___global___geostatinversion___xis))
	end
	result = pcgalm(forwardmodel, s0, X, length(xis), getmodelparamsshort, Rdiag, y; maxiters=maxiters, delta=delta, xtol=xtol)
	___global___geostatinversion___xis = nothing
	return result
	#=
	function lm_f(x::Vector)
		modelparams = getmodelparams(x, X, xis)
		modelpredictions = forwardmodel(modelparams)
		return sqrt(Rdiag) .* (modelpredictions - y)
	end
	lm_g = FDDerivatives.makejacobian(lm_f)
	opt = Optim.levenberg_marquardt(lm_f, lm_g, [zeros(length(xis)); 1.])
	result = getmodelparams(opt.minimum, X, xis)
	return result
	=#
end

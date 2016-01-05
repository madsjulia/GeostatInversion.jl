function getmodelparams(x, X, xis)
	modelparams = X * x[end]
	for i = 1:length(xis)
		BLAS.axpy!(x[i], xis[i], modelparams)
	end
	return modelparams
end

function pcgalm(forwardmodel::Function, s0::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, Rdiag::Vector, y::Vector; maxiters::Int=5, delta::Float64=sqrt(eps(Float64)), xtol::Float64=1e-6)
	function lm_f(x::Vector)
		modelparams = getmodelparams(x, X, xis)
		modelpredictions = forwardmodel(modelparams)
		return sqrt(Rdiag) .* (modelpredictions - y)
	end
	lm_g = FDDerivatives.makejacobian(lm_f)
	results = Optim.levenberg_marquardt(lm_f, lm_g, [zeros(length(xis)); 1.])
	return getmodelparams(results.minimum, X, xis)
end

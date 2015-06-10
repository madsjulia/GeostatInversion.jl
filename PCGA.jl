module PCGA

const delta = sqrt(eps())

function pcgaiteration(forwardmodel, s, X, xis, R, v)
	global delta
	p = 1
	K = length(xis)
	m = length(xis[1])
	paramstorun = [s; s + delta * xis; s + delta * X; s + delta * s]
	results = pmap(forwardmodel, paramstorun)
	n = length(results[1])
	etai = Array(Float64, m)
	HQHpR = R
	HQ = zeros(n, m)
	for i = 1:K
		etai = (results[i + 1] - results[1]) / delta
		HQ += etai * transpose(xis[i])
		HQH += etai * transpose(etai)
	end
	HX = (results[end - 1] - results[1]) / delta
	Hs = (results[end] - results[1]) / delta
	A = [HQHpR HX; transpose(HX) zeros(p, p)]
	b = [v - results[1] + Hs; zeros(p)]
	x = A \ b
	s_new = X * x[end] + transpose(HQ) * x[1:end-1]
	return s_new
end

end

import GeostatInversion
import RMF
using Base.Test

function setupsimpletest(M, N)
	x = randn(N)
	Q0 = randn(M, N)
	Q = Q0' * Q0
	sqrtQ = sqrtm(Q)
	truep = real(sqrtQ * randn(N))
	function forward(p::Vector)
		return p .* x
	end
	truey = forward(truep)
	xis = Array(Array{Float64, 1}, round(Int, M))
	Z = RMF.randsvd(Q, length(xis), round(Int, 0.1 * M), 3)
	for i = 1:length(xis)
		xis[i] = Z[:, i]
	end
	X = zeros(N)
	noiselevel = 0.0001
	R = noiselevel ^ 2 * eye(N)
	yobs = truey + noiselevel * randn(N)
	p0 = zeros(N)
	return forward, p0, X, xis, R, yobs, truep
end

function simpletestpcga(M, N)
	forward, p0, X, xis, R, yobs, truep = setupsimpletest(M, N)
	popt = GeostatInversion.pcga(forward, p0, X, xis, R, yobs)
	@test_approx_eq_eps norm(popt - truep) / norm(truep) 0. 1e-2
end

function simpletestrga(M, N, Nreduced)
	forward, p0, X, xis, R, yobs, truep = setupsimpletest(M, N)
	S = randn(Nreduced, N) * (1 / sqrt(N))
	popt = GeostatInversion.rga(forward, p0, X, xis, R, yobs, S)
	@test_approx_eq_eps norm(popt - truep) / norm(truep) 0. 1e-2
end

srand(0)
maxlog2N = 8
minlog2N = 2
for log2N = minlog2N:maxlog2N
	for log2M = 0:log2N - 1
		N = 2 ^ log2N
		M = 2 ^ log2M
		simpletestpcga(M, N)
		simpletestrga(M, N, 2 ^ (log2N - 1))#test reducing the observations by a factor of 2
	end
end

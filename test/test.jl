import GeostatInversion
import RMF
import FFTRF
using Base.Test

function simplepcgalowranktest(numetas=10, numobs=20)
	#[(HQH + R) HX; transpose(HX) zeros(p, p)]
	noiselevels = [1e16, 0.]#none, nuge
	etagenerators = [zeros, randn]
	HXgenerators = [zeros, randn]
	etas = Array(Array{Float64, 1}, numetas)
	for noiselevel in noiselevels
		for etagenerator in etagenerators
			for HXgenerator in HXgenerators
				HQH = zeros(numobs, numobs)
				for i = 1:numetas
					etas[i] = etagenerator(numobs)
					BLAS.ger!(1., etas[i], etas[i], HQH)
				end
				HX = HXgenerator(numobs)
				R = noiselevel * speye(numobs)
				bigA = [(HQH + R) HX; transpose(HX) zeros(1, 1)]
				lrbigA = GeostatInversion.PCGALowRankMatrix(etas, HX, R)
				for i = 1:numobs + 1
					x = zeros(numobs + 1)
					x[i] = 1.
					@test_approx_eq bigA * x lrbigA * x
				end
			end
		end
	end
end

function simplelowrankcovtest()
	samples = Array{Float64, 1}[[-.5, 0., .5], [1., -1., 0.], [-.5, 1., -.5]]
	lrcm = GeostatInversion.LowRankCovMatrix(samples)
	fullcm = eye(3) * lrcm
	@test_approx_eq fullcm lrcm * eye(3)
	@test_approx_eq sum(map(x->x * x', samples)) / (length(samples) - 1) fullcm
	@test_approx_eq sum(map(x->x * x', samples)) / (length(samples) - 1) fullcm
	for i = 1:100
		x = randn(3, 3)
		@test_approx_eq fullcm * x lrcm * x
		@test_approx_eq fullcm' * x lrcm' * x
	end
end

function lowrankcovconsistencytest()
	const N = 10000
	const M = 100
	sqrtcovmatrix = randn(M, M)
	covmatrix = sqrtcovmatrix * sqrtcovmatrix'
	samples = Array(Array{Float64, 1}, N)
	onesamples = Array(Float64, N)
	twosamples = Array(Float64, N)
	for i = 1:N
		samples[i] = sqrtcovmatrix * randn(M)
		onesamples[i] = samples[i][1]
		twosamples[i] = samples[i][2]
	end
	lrcm = GeostatInversion.LowRankCovMatrix(samples)
	lrcmfull = lrcm * eye(M)
	@test_approx_eq_eps norm(lrcmfull - covmatrix, 2) 0. M ^ 2 / sqrt(N)
end


function lowrankcovgetxistest()
	numfields = 100
	numxis = 30
	p = 20
	samplefield() = FFTRF.powerlaw_structuredgrid([25, 25], 2., 3.14, -3.5)[1:end]
	srand(0)
	lrcmxis = GeostatInversion.getxis(samplefield, numfields, numxis, p)
	srand(0)
	fields = Array(Array{Float64, 1}, numfields)
	for i = 1:numfields
		fields[i] = samplefield()
	end
	lrcm = GeostatInversion.LowRankCovMatrix(fields)
	fullcm = eye(size(lrcm, 1)) * lrcm
	fullxis = GeostatInversion.getxis(fullcm, numxis, p)
	for i = 1:length(fullxis)
		#=
		Apparently due to minor discrepancies (rounding error), the LU decomposition is not
		consistently reproducible. As a consequence, the randsvd part of the getxis can
		return + or - the singular vectors. We check that it is close to + or - the xis
		we get from the full matrix.
		=#
		@test_approx_eq_eps 0. min(norm(fullxis[i] - lrcmxis[i]), norm(fullxis[i] + lrcmxis[i])) 1e-6
	end
end

function setupsimpletest(M, N, mu)
	x = randn(N)
	Q0 = randn(M, N)
	Q = Q0' * Q0
	sqrtQ = sqrtm(Q)
	truep = real(sqrtQ * randn(N)) + mu
	function forward(p::Vector)
		return p .* x
	end
	truey = forward(truep)
	xis = GeostatInversion.getxis(Q, M, round(Int, 0.1 * M))
	X = fill(mu, N)
	noiselevel = 0.0001
	R = noiselevel ^ 2 * speye(N)
	yobs = truey + noiselevel * randn(N)
	p0 = fill(mu, N)
	return forward, p0, X, xis, R, yobs, truep
end

function simpletestpcga(M, N, mu=0.)
	forward, p0, X, xis, R, yobs, truep = setupsimpletest(M, N, mu)
	println("$M, $N")
	print("Direct:")
	@time popt = GeostatInversion.pcgadirect(forward, p0, X, xis, R, yobs)
	@test_approx_eq_eps norm(popt - truep) / norm(truep) 0. 2e-2
	if M < N / 6
		print("LSQR:")
		@time popt = GeostatInversion.pcgalsqr(forward, p0, X, xis, R, yobs)
		@test_approx_eq_eps norm(popt - truep) / norm(truep) 0. 2e-2
	end
	println()
end

function simpletestrga(M, N, Nreduced, mu=0.)
	forward, p0, X, xis, R, yobs, truep = setupsimpletest(M, N, mu)
	S = randn(Nreduced, N) * (1 / sqrt(N))
	popt = GeostatInversion.rga(forward, p0, X, xis, R, yobs, S)
	@test_approx_eq_eps norm(popt - truep) / norm(truep) 0. 2e-2
end

function simpletestpcgalm(M, N, mu=0.)
	forward, p0, X, xis, R, yobs, truep = setupsimpletest(M, N, mu)
	println("$M, $N")
	print("LM:")
	@time popt = GeostatInversion.pcgalm(forward, p0, X, xis, diag(R), yobs)
	@test_approx_eq_eps norm(popt - truep) / norm(truep) 0. 2e-2
end

srand(0)
simplepcgalowranktest()
simplelowrankcovtest()
lowrankcovconsistencytest()
lowrankcovgetxistest()
simpletestrga(2 ^ 3, 2 ^ 10, 2 ^ 9)
simpletestrga(2 ^ 3, 2 ^ 11, 2 ^ 9)
simpletestrga(2 ^ 3, 2 ^ 10, 2 ^ 9, 10.)
simpletestrga(2 ^ 3, 2 ^ 11, 2 ^ 9, 10.)
maxlog2N = 8
minlog2N = 2
for log2N = minlog2N:maxlog2N
	for log2M = 0:log2N - 1
		N = 2 ^ log2N
		M = 2 ^ log2M
		simpletestpcga(M, N)
		simpletestpcga(M, N, 10.)
		simpletestpcgalm(M, N)
		simpletestpcgalm(M, N, 10.)
	end
end

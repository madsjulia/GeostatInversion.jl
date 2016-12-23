using Base.Test
import GeostatInversion
const FFTRF = GeostatInversion.FFTRF

function testNd(N)
	Ns = map(x->round(Int, 25 * x), 1 + rand(N))
	k0 = randn()
	dk = rand()
	beta = -2 - rand()
	k = FFTRF.powerlaw_structuredgrid(Ns, k0, dk, beta)
	@test_approx_eq mean(k) k0
	@test_approx_eq std(k) dk
	@test collect(size(k)) == Ns
end

function testunstructured(N)
	points = randn(N, 100)
	Ns = map(x->round(Int, 25 * x), 1 + rand(N))
	k0 = randn()
	dk = rand()
	beta = -2 - rand()
	k = FFTRF.powerlaw_unstructuredgrid(points, Ns, k0, dk, beta)
	@test length(k) == size(points, 2)
end

srand(0)
for i = 1:10
	testunstructured(2)
	testunstructured(3)
	testNd(2)
	testNd(3)
end


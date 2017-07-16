import GeostatInversion
import Base.Test

@stderrcapture function testNd(N)
	Ns = map(x->round(Int, 25 * x), 1 + rand(N))
	k0 = randn()
	dk = rand()
	beta = -2 - rand()
	k = GeostatInversion.FFTRF.powerlaw_structuredgrid(Ns, k0, dk, beta)
	@Base.Test.test mean(k) ≈ k0
	@Base.Test.test std(k) ≈ dk
	@Base.Test.test collect(size(k)) == Ns
end

@stderrcapture function testunstructured(N)
	points = randn(N, 100)
	Ns = map(x->round(Int, 25 * x), 1 + rand(N))
	k0 = randn()
	dk = rand()
	beta = -2 - rand()
	k = GeostatInversion.FFTRF.powerlaw_unstructuredgrid(points, Ns, k0, dk, beta)
	@Base.Test.test length(k) == size(points, 2)
end

srand(2017)
@Base.Test.testset "FTRF" begin
	for i = 1:10
		testunstructured(2)
		testunstructured(3)
		testNd(2)
		testNd(3)
	end
end


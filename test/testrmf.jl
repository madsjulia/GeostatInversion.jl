import GeostatInversion
const RandMatFact = GeostatInversion.RandMatFact
using Base.Test

function makeA(n, m)
	range = randn(n, m)
	other = randn(m, n)
	return range * other
end

function test_rangefinder(n, m)
	A = makeA(n, m)
	Q = RandMatFact.rangefinder(A)
	@test_approx_eq_eps m size(Q, 2) 1
	@test_approx_eq_eps vecnorm(A - Q * ctranspose(Q) * A) 0. 1e-8
	Q = RandMatFact.rangefinder(A, m, 2)
	@test_approx_eq_eps m size(Q, 2) 1
	@test_approx_eq_eps vecnorm(A - Q * ctranspose(Q) * A) 0. 1e-8
end

function test_eig_nystrom()
	A = Float64[2 -1 0; -1 2 -1; 0 -1 2]
	Q = RandMatFact.rangefinder(A)
	U, Sigmavec = RandMatFact.eig_nystrom(A, Q)
	Sigma = diagm(Sigmavec)
	Lambda = Sigma * Sigma
	eigvals, eigvecs = eig(A)
	@test_approx_eq_eps vecnorm(sort(eigvals, rev=true) - diag(Lambda)) 0. 1e-8
end

test_rangefinder(10, 2)
test_rangefinder(10, 5)
test_rangefinder(100, 5)
test_rangefinder(100, 10)
test_rangefinder(100, 25)
test_eig_nystrom()
#=
#some tests for scaling performance
@time A = makeA(1000, 20)
@time Q = RandMatFact.rangefinder(A)
nothing
=#

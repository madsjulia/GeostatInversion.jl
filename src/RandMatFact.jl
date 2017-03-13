"Random Matrix Factorization Functions"
module RandMatFact

function colnorms(Y)
	norms = Array{Float64}(size(Y, 2))
	for i = 1:size(Y, 2)
		norms[i] = norm(Y[:, i])
	end
	return norms
end

function rangefinder(A; epsilon=1e-8, r=10)#implements algorithm 4.2 in halko et al
	m = size(A, 1)
	n = size(A, 2)
	Yfull = zeros(Float64, n, r + min(n, m))
	Y = view(Yfull, :, 1:r)
	BLAS.gemm!('N', 'N', 1., A, randn(n, r), 0., Y)
	omega = Array{Float64}(n)
	j = 0
	Qfull = zeros(Float64, m, min(n, m))
	tempvec = Array{Float64}(m)
	Aomega = Array{Float64}(m)
	while maximum(colnorms(view(Yfull, :, j+1:j+r))) > epsilon / sqrt(200 / pi)
		j = j + 1
		Yj = view(Yfull, :, j)
		Q = view(Qfull, :, 1:j - 1)
		QtYj = BLAS.gemv('T', 1., Q, Yj)
		Yj -= Q * QtYj
		q = Yj / norm(Yj)
		Qj = view(Qfull, :, j)
		BLAS.axpy!(1 / norm(Yj), Yj, Qj)
		Q = view(Qfull, :, 1:j)
		randn!(omega)
		Aomega = BLAS.gemv!('N', 1., A, omega, 0., Aomega)
		QtAomega = BLAS.gemv('T', 1., Q, Aomega)
		ynew = Aomega - Q * QtAomega
		Yfull[:, r + j] = ynew
		Qj = view(Qfull, :, j)
		for i = j + 1:j + r - 1
			Yi = view(Yfull, :, i)
			BLAS.axpy!(-dot(Qj, Yi), Qj, Yi)
		end
	end
	return Qfull[:, 1:j]
end

function rangefinder(A, l::Int64, numiterations::Int64)
	#TODO rewrite this to use qrfact! and lufact! to save on memory (we are always overwriting Q anyway)
	m = size(A, 1)
	n = size(A, 2)
	Omega = randn(n, l) #Gaussian requires less oversampling but is more costly to construct, see sect 4.6 Halko
	Y = A * Omega
	if numiterations == 0
		Q, R, () = qr(Y, Val{true})#pivoted QR is more numerically stable
		return Q
	elseif numiterations > 0
		Q, R = lu(Y)
	else
		error("parameter numiterations should be positive, but numiterations=$numiterations")
	end
	#Conduct normalized power iterations.
	for i = 1:numiterations
		Q = A' * Q
		Q, R = lu(Q)
		Q = A * Q
		if i < numiterations
			Q, R = lu(Q)
		else
			Q, R, () = qr(Q, Val{true})
		end
	end
	return Q
end

"Random SVD based on algorithm 5.1 from Halko et al."
function randsvd(A, K::Int, p::Int, q::Int)
	Q = rangefinder(A, K + p, q);
	B = Q' * A;
	(), S, V = svd(B);#This is algorithm 5.1 from Halko et al, Direct SVD
	Sh = diagm(sqrt([S[1:K]; zeros(p)]))#Cut back to K from K+p
	Z = V * Sh
	return Z
end

function eig_nystrom(A, Q)#implements algorithm 5.5 from Halko et al
	B1 = A * Q
	B2 = ctranspose(Q) * B1
	C = chol(Hermitian(B2))
	F = B1 * inv(C)#this should be replaced by triangular solve if it is slowing things down
	U, Sigmavec, V = svd(F)
	#Sigma = diagm(Sigmavec)
	#Lambda = Sigma * Sigma
	#return U, Lambda
	return U, Sigmavec
end

end

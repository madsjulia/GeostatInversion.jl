#module PCGA
#Definetly not array optimized
#using Debug

const delta = sqrt(eps())

# Inputs: 
# forwardmodel - param to obs map
#            s - current iterate          
#            X - mean of parameter prior (replace with B*X drift matrix
# later for p>1)
#          xis - K columns of Z where Q approx= ZZ^T
#            R - covariance of measurement error (data misfit term)
#            y - data vector            


function colnorms(Y)
	norms = Array(Float64, size(Y, 2))
	for i = 1:size(Y, 2)
		norms[i] = norm(Y[:, i])
	end
	return norms
end

function rangefinder(A; epsilon=1e-8, r=10)#implements algorithm 4.2 in halko et al
	m = size(A, 1)
	n = size(A, 2)
	Omega = randn(n, r)
	Y = A * Omega
	j = 0
	Q = Array(Float64, m, 0)
	while max(colnorms(Y[:, j+1:j+r])...) > epsilon / sqrt(200 / pi)
		j = j + 1
		Y[:, j] = (eye(m) - Q * ctranspose(Q)) * Y[:, j]
		q = Y[:, j] / norm(Y[:, j])
		Q = [Q q]
		Omega = [Omega randn(n)]
		ynew = (eye(m) - Q * ctranspose(Q)) * A * Omega[:, j + r]
		Y = [Y ynew]
		for i = j + 1:j + r - 1
			Y[:, i] = Y[:, i] - Q[:, j] * dot(Q[:, j], Y[:, i])
		end
	end
	return Q
end

function randSVD(A; epsilon=1e-8, r=10)
Q = rangefinder(A; epsilon=1e-8, r=10);
B = Q' * A;
Util,() = svd(B);
U = Q*Util;
return U
end 

include("deconvolutionTestProblem.jl")
A,strue,yvec,Gamma,C = deconv2(20,5);

U = randSVD(C);

# @debug function pcgaiteration(s, X, xis, R, y; forwardmodel = testForward)
function pcgaiteration(forwardmodel,s, X, xis, R, y)

    global delta
    p = 1
    K = length(xis)
    m = length(xis[1])
    paramstorun = [s .+ delta .* xis,{s,s + delta * X, s + delta * s}]
    results = pmap(forwardmodel, paramstorun)
    n = length(results[1])
    # etai = Array(Float64, m)   
    HQHpR = R
    HQ = zeros(n, m)
    for i = 1:K
	etai = (results[i] - results[K+1]) / delta
# @bp
	HQ += etai * transpose(xis[i])
	HQHpR += etai * transpose(etai)
    end
    HX = (results[end - 1] - results[1]) / delta
    Hs = (results[end] - results[K+1]) / delta
    A = [HQHpR HX; transpose(HX) zeros(p, p)]
    b = [y - results[1] + Hs; zeros(p)]
    x = A \ b
    s_new = X * x[end] + transpose(HQ) * x[1:end-1]
    return s_new, HQ,HQHpR
end

 Xis = {U[:,1],U[:,2]};

for i = 3:size(U,2)
     Xis = push!(Xis,U[:,i])
end

#s0 = 0.5*ones(length(strue));
s0 = strue;
s1,HQ,HQHpR = pcgaiteration(testForward, s0, strue, Xis, Gamma, yvec)

# this doesn't work need to check formation of etas
#  norm(HQHpR-(A'*C*A+Gamma))
# 2.059780712550066e18


#end

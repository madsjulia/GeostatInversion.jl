# Ellen Le
# July 7, 2015 LANL
# Checks the accuracy of the approximations for Hs, HX, H'QHpR, and HQ

using PyPlot
const noise = 4
const numparams = 30
const delta = sqrt(eps())

include("deconvolutionTestProblem.jl")

H,strue,yvec,R,Q = deconv2(numparams,noise);

function testForward(u)
    y = H*u;
    return y
end

## 1. Test Hs, linearized operator acting on current iterate
s0 = vec(zeros(numparams,1)) #change to the prior mean

#choose a random smooth field in the prior
U,S = svd(Q) #assuming Q not perfectly spd
Sh = sqrt(S)
L = U*diagm(Sh)
srand(1234)
s_rand = s0 +  L * randn(numparams,1);

x = linspace(0,1,numparams);
plot(x,s_rand,x,strue)

#check
checkL = norm(L*L' - Q)/norm(Q)

s_syn = strue
#Hs = (H*(s0+delta*s0)-H(s0))/delta, note linear so it simplifies
Hs = H*s_rand
Hs_syn = H*s_syn
approxHs = (testForward(s_rand + delta*s_rand)-testForward(s_rand))/delta
approxHs_syn = (testForward(s_syn+delta*s_syn)-testForward(s_syn))/delta

relnormHs = norm(approxHs - Hs)/norm(Hs) #will be zero since linear
relnormHs_synthetic = norm(approxHs_syn - Hs_syn)/norm(Hs_syn) #this is just the error
#in matrix multiplication, both should be on the order of delta 1.5e-8

## 2. Test HX

# This is meaningless for the linearized operator unless we assume a
# nonzero mean, but see section 1 above for some other checks

## 3. Test H'QHpR and HQ

# Form the etas which are really H*zeta
# Z_i = zeta_i then Q = Z*Z'
# First we find and approx W st Q = W*W'*Q using rangefinder with
# oversampling r
# Then do an SVD of the smaller matrix W'*Q, and get our approximate
# eigendecomposition Q = W*U*S*V', assuming that approximately WU=V
# Then Z = WU*diagm(sqrt(S)) or = V*diagm(sqrt(S))
# We find that Z1 = W*U is terrible and not at all similar to V contrary
# to p 5425

function colnorms(Y)
    norms = Array(Float64, size(Y, 2))
    for i = 1:size(Y, 2)
	norms[i] = norm(Y[:, i])
    end
    return norms
end

function rangefinder(A; epsilon=1e-10, r=20)#implements algorithm 4.2 in halko et al
    m = size(A, 1)
    n = size(A, 2)
    Omega = randn(n, r)
    Y = A * Omega
    j = 0
    W = Array(Float64, m, 0)
    while max(colnorms(Y[:, j+1:j+r])...) > epsilon / sqrt(200 / pi)
	j = j + 1
	Y[:, j] = (eye(m) - W * ctranspose(W)) * Y[:, j]
	q = Y[:, j] / norm(Y[:, j])
	W = [W q]
	Omega = [Omega randn(n)]
	ynew = (eye(m) - W * ctranspose(W)) * A * Omega[:, j + r]
	Y = [Y ynew]
	for i = j + 1:j + r - 1
	    Y[:, i] = Y[:, i] - W[:, j] * dot(W[:, j], Y[:, i])
	end
    end
    return W
end

function randSVD(A; epsilon=1e-10, r=20)
    W = rangefinder(A; epsilon=1e-10, r=20);
    B = W' * A;
    Util,S,V = svd(B);
    U = Q*Util;
    return U,S,V,W
end  

U,S,V,W = randSVD(Q)
Sh = diagm(sqrt(S))
Z1 = W*U*Sh
Z2 = V*Sh

rel_errRangeFind = norm(Q - W*W'*Q)/norm(Q)
rel_errZ1 = norm(Q-Z1*Z1')/norm(Q) 
rel_errZ2 = norm(Q-Z2*Z2')/norm(Q)

#Note that Z2 is much better, so we take zeta_i = Z2_i

Z = Z2

# The Zis are saved before we start our iteration
Zis = Array{Float64, 1}[Z[:,1],Z[:,2]];

for i = 3:size(Z,2)
    Zis = push!(Zis,Z[:,i])
end

Eta_matrix = H * Z #These change at each iteration if the forward solver
#is nonlinear, since H is the linearization at the current point

HQ_exact = H*Q
HQHpR_exact = H*Q*H' + R

function pcgaiteration(forwardmodel::Function,s::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R::Matrix, y::Vector)
    # Inputs: 
    # forwardmodel - param to obs map h(s)
    #            s - current iterate s_k or sbar          
    #            X - mean of parameter prior (replace with B*X drift matrix
    # later for p>1)
    #          xis - K columns of Z where Q approx= ZZ^T, get this by doing
    #          random SVD on your prior covariance matrix and save the
    #          columns in a list xis = [col1,col2,....]
    #            R - covariance of measurement error (data misfit term)
    #            y - data vector     

    global delta
    p = 1
    K = length(xis)
    m = length(xis[1])
    #paramstorun = Array{Float64, 1}[s .+ delta .* xis,{s,s + delta * X, s + delta * s}]
	paramstorun = Array(Array{Float64, 1}, length(xis) + 3)
	for i = 1:length(xis)
		paramstorun[i] = s + delta * xis[i]
	end
	paramstorun[length(xis) + 1] = s
	paramstorun[length(xis) + 2] = s + delta * X
	paramstorun[length(xis) + 3] = s + delta * s
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
    bigA = [HQHpR HX; transpose(HX) zeros(p, p)]
    b = [y - results[1] + Hs; zeros(p)]
    x = bigA \ b # we will replace this with a Krylov solver or something
rele    # like UMFPACK?
    s_new = X * x[end] + transpose(HQ) * x[1:end-1]
	#println(s_new - s)
    return HQ,HQHpR
end

# Cheating, we use the synthetic as the mean
HQ,HQHpR = pcgaiteration(testForward, s0, strue, Zis, R, yvec)

rel_errHQ = norm(HQ_exact-HQ)/norm(HQ_exact);
rel_errHQHpR = norm(HQHpR_exact-HQHpR)/norm(HQHpR_exact);

@show(rel_errRangeFind)
@show(rel_errZ1)
@show(rel_errZ2)
@show(rel_errHQ)
@show(rel_errHQHpR)









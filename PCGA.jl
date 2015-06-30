#module PCGA
#Definetly not array optimized
#using Debug
using PyPlot


const EXAMPLEFLAG = 1 # 1 = deconvolution test problem
# 2 = ellen.jl forward model

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

if EXAMPLEFLAG == 1
    include("deconvolutionTestProblem.jl")
    A,strue,yvec,Gamma,C = deconv2(20,5);
    U = randSVD(C); #Do random SVD on the prior part covariance matrix
elseif EXAMPLEFLAG == 2
    include("~/codes/finitedifference2d.jl/ellen.jl")
else
    println("example not supported")
end


# @debug function pcgaiteration(s, X, xis, R, y; forwardmodel = testForward)
function pcgaiteration(forwardmodel,s, X, xis, R, y)
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
    x = A \ b # we will replace this with a Krylov solver or something
    # like UMFPACK?
    s_new = X * x[end] + transpose(HQ) * x[1:end-1]
    return s_new #, HQ,HQHpR
end

Xis = {U[:,1],U[:,2]};

for i = 3:size(U,2)
    Xis = push!(Xis,U[:,i])
end


#Runs the optimization loop until it converges
const total_iter = 102;
#s0 = strue+0.05*randn(length(strue));
s0 = 0.5*ones(length(strue));
relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

for k = 1:total_iter
    sbar[:,k+1] = pcgaiteration(testForward, sbar[:,k], strue, Xis, Gamma, yvec)
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end

return sbar,relerror

# this doesn't work need to check formation of etas
#  norm(HQHpR-(A'*C*A+Gamma))
# 2.059780712550066e18

x = linspace(0,1,20);
plot(x,strue,x,sbar[:,1],x,sbar[:,end-1],x,sbar[:,end],linestyle="-",marker="o")
legend(["sythetic","initial s_0","s_end-1","s_end"])
xlabel("unit 1D domain x")
ylabel("1D parameter field s(x)")
title(["Various iterates, total iterates = ",repr(total_iter)])
grid("on")

figure(2)
plot(1:total_iter+1,relerror,linestyle="-",marker="o")
title("Relative error vs iteration number, PCGA method")

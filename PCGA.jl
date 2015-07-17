module PCGA
using Debug

# Julia implementation of PCGA, a method for solving the
# subsurface inverse problems using a randomized low rank approximation of
# the prior covariance, and a finite difference approximation of the
# gradient. No adjoints, gradients, or Hessians required.  
# Last updated July 17, 2015 by Ellen Le
# Questions: ellenble@gmail.com

# References: 
# Jonghyun Lee and Peter K. Kitanidis, 
# Large-Scale Hydraulic Tomography and Joint Inversion of Head and
# Tracer Data using the Principal Component Geostatistical Approach
# (PCGA), 
# Water Resources Research, 50(7): 5410-5427, 2014
# Peter K. Kitanidis and Jonghyun Lee, 
# Principal Component Geostatistical Approach for Large-Dimensional
# Inverse Problem, 
# Water Resources Research, 50(7): 5428-5443, 2014

const delta = sqrt(eps())

function colnorms(Y)
    norms = Array(Float64, size(Y, 2))
    for i = 1:size(Y, 2)
	norms[i] = norm(Y[:, i])
    end
    return norms
end

function rangefinder(A::Matrix,l::Int64,its::Int64)
    srand(1)
    m = size(A, 1)
    n = size(A, 2)
    Omega = randn(n, l) #Gaussian requires less oversampling but is more
    #costly to construct, see sect 4.6 Halko
    Y = A*Omega

    if its == 0
        Q,R,() = qr(Y,pivot = true); #pivoted QR is more numerically
        #stable
    end
    
    if its > 0
        Q,R = lu(Y)
    end

#   Conduct normalized power iterations.
#
    for it = 1:its

      Q = (Q'*A)';

      Q,R = lu(Q);

      Q = A*Q;

      if it < its
        Q,R = lu(Q);
      end

      if it == its
        Q,R,() = qr(Q,pivot = true);
      end

    end

    return Q
end

function randSVDzetas(A::Matrix,K::Int64,p::Int64,q::Int64)
    Q = rangefinder(A,K+p,q);
    B = Q' * A;      # 
    (),S,V = svd(B); #This is algorithm 5.1, Direct SVD
    Sh = diagm(sqrt([S[1:K];zeros(p)])) # Cut back to K from K+p
    Z = V*Sh 
    return Z
end  

function pcgaiteration(forwardmodel::Function,s::Vector, X::Vector, xis::Array{Array{Float64, 1}, 1}, R::Matrix, y::Vector)
    # Inputs: 
    # forwardmodel - param to obs map h(s)
    #            s - current iterate s_k or sbar          
    #            X - mean of parameter prior (replace with B*X drift matrix
    # later for p>1)
    #          xis - K columns of Z = randSVDzetas(Q,K,p,q) where Q approx= ZZ^T
    #            R - covariance of measurement error (data misfit term)
    #            y - data vector
    global delta
    p = 1
    K = length(xis)
    m = length(xis[1])
    paramstorun = Array(Array{Float64, 1}, length(xis) + 3)
    for i = 1:length(xis)
	paramstorun[i] = s + delta * xis[i]
    end
    paramstorun[length(xis) + 1] = s
    paramstorun[length(xis) + 2] = s + delta * X
    paramstorun[length(xis) + 3] = s + delta * s
    results = pmap(forwardmodel, paramstorun)
    n = length(results[1])
    HQHpR = R
    HQ = zeros(n, m)
    for i = 1:K
	etai = (results[i] - results[K+1]) / delta
       	HQ += etai * transpose(xis[i])
	HQHpR += etai * transpose(etai)
    end
    HX = (results[end-1] - results[end-2]) / delta
    Hs = (results[end] - results[K+1]) / delta
    bigA = [HQHpR HX; transpose(HX) zeros(p, p)]
    b = [y - results[1] + Hs; zeros(p)]
    x = pinv(bigA) * b
    s_new = X * x[end] + (HQ)'* x[1:end-1]
    return s_new 
end

end

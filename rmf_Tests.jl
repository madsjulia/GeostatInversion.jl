using HDF5, JLD

SAVEDQ_FLAG = 2

# Implements and tests random matrix factorization techniques in Halko
# et al, modified for PCGA.
# Some code stolen from Mark Tygert's Matlab implementation of
# randomized PCA
# Last updated July 17, 2015 by Ellen Le
#
#   Reference:
#   Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
#   Finding structure with randomness: probabilistic algorithms
#   for constructing approximate matrix decompositions,
#   arXiv:0909.4061 [math.NA; math.PR], 2009
#   (available at http://arxiv.org).

function rangefinder1(A,l,its)
# Finds an orthonormal basis with l columns to approx. range of A - 
# Algo 4.1 p 240 in Halko
# Use for when you want to specify the rank of the decomposition. 
# For specifying the accuracy, use Algo 4.2 instead
# If you want a rank k, choose l = k + p where p = 5,10,15
# To do power iterations, change q, otherwise set q = 0
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

srand(1)
#Creates a random nxn input matrix A of rank m. Tests Algo 4.1 with oversampling p.
function test_rangefinder(n, m)
    A = Array(Float64, (n, n))
    range = randn(n, m)
    for i = 1:n
	sum = zeros(n)
	for j = 1:m
	    sum += randn() * range[:, j]
	end
	A[:, i] = sum
    end
    tic()
    Q1 = rangefinder1(A,m+10)
    saveTime = toq()
    println("4.1: Rank of A is $(m) and should be less than or equal to the no. of columns of Q which is $(size(Q1, 2))")
    println("4.1: Stage A error $(norm(A - Q1 * ctranspose(Q1) * A)) should be less than desired tolerance")
    println("4.1: time for this algorithm is $(saveTime) seconds")
end

## Below are test to see what rank K is optimal for the 2D problem
lenCoords = 840

# Load the Q from the 2D problem
if SAVEDQ_FLAG == 1
    Q = load("pcga.jl/ellenQ.jld","Q");
    Z = load("pcga.jl/ellenQ.jld","Z");
elseif SAVEDQ_FLAG == 2
    include("ellen.jl")
else
    println("check Q")
end
@show(rank(Q))

p = 20; q=0; #change p and q here and in the line below
function randSVDzetas(A,K; p = 20, q=0)
    Q = rangefinder1(A,K+p,q);
    B = Q' * A;      # 
    U,S,V = svd(B); #This is algorithm 5.1, Direct SVD
    Sh = diagm(sqrt([S[1:K];zeros(p)])) #Oversample by p
    Z = V*Sh 
    return Z
end  

ranks = [50:25:300]
n_ranks = length(ranks)

relerr_a = Array(Float64,n_ranks)
relerr_b = Array(Float64,n_ranks)

# The best rank-100 approximation (deterministic), save this SVD for plot
lenQ = size(Q,1)
tic()
U,S,V = svd(Q)
Sh = sqrt(S)
Z1 = V*diagm([Sh[1:100];zeros(lenQ-100)])
time_bestrankK = toq()

# Plot error comparison of randomized rank-K versus best rank-K
for i = 1:n_ranks
    K = ranks[i]
    Za = randSVDzetas(Q,K)
    relerr_a[i] = norm(Q-Za*Za')/norm(Q)

    Sh = sqrt(S)
    Zb = V*diagm([Sh[1:K];zeros(lenQ-K)])
    relerr_b[i] = norm(Q-Zb*Zb')/norm(Q)
end

using PyPlot
figure()
plot(ranks,relerr_a,ranks,relerr_b,marker="o")
xlabel("RANK (of basis for approximate range of Q and approx rank of decomposition)")
ylabel("RELATIVE ERROR ||Q-Z*Z'||/||Q||")
title("Rank of approximation versus error with p=$(p), q=$(q)")
legend(["approximation error with p=$(p),q=$(q) ", "true relative error of best rank k
        approx"],loc = "best")

tic()
Z2 = randSVDzetas(Q,100)
time_approxrankK = toq()

relerror_bestrankK = norm(Q-Z1*Z1')/norm(Q)
relerror_approxrankK = norm(Q-Z2*Z2')/norm(Q)

@show(p,q,rank(Z1),relerror_bestrankK,time_bestrankK,rank(Z2),relerror_approxrankK,time_approxrankK)
timespeedup = time_bestrankK/time_approxrankK
@show(timespeedup)


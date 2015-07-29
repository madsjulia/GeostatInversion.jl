import PCGA
import Krigapping

const EXAMPLEFLAG = 2 
const CASEFLAG = 6

# Driver for tests using module PCGA.jl. 2 examples available.
# Set:
# EXAMPLEFLAG = 1  for 1D deconvolution test problem in deconvolutionTestProblem.jl
# EXAMPLEFLAG = 2  for 2D groundwater forward model in ellen.jl
# CASEFLAG = 1     for mean_s = 0. and random starting point s0 using the prior
# CASEFLAG = 2     for mean_s = 0.3. and s0 = 0.6 (homogeneous)
# CASEFLAG = 3     for mean_s = 0.3. and random starting point s0 using the prior
# CASEFLAG = 4     for mean_s = pertrubed truth and s0 = 0.3.(homogeneous)
# CASEFLAG = 5     for mean_s = 0. and s0 = 0.
# CASEFLAG = 6     for mean_s = s0 = kriging of noisy obs points using Q

# Last updated July 17, 2015 by Ellen Le
# Questions: ellenble@gmail.com
#
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

#Run the optimization loop until it converges or a total_iter number of times
const total_iter = 2;

if EXAMPLEFLAG == 1
    using PyPlot
    include("deconvolutionTestProblem.jl")
    const noise = 5  #  noise = 5 means 5% of max value
    const numparams = 30
    G,strue,yvec,Gamma,Q = deconv2(numparams,noise);

    const p = 0 # The oversampling parameter for increasing RSVD accuracy
    const q = 0 # Number of power iterations in the RSVD
    const K = 10 # Set the rank of the RSVD for Q, take i.e. K =
                # ceil(length(strue)/7) 

    Z = PCGA.randSVDzetas(Q,K,p,q); # Random SVD on the prior part covariance matrix
elseif EXAMPLEFLAG == 2
    include("ellen.jl")
    testForward = forwardObsPoints
    Gamma = R
    strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field
    yvec = u_obsNoise # see ellen.jl for noise level

    const p = 20 # The oversampling parameter for increasing RSVD accuracy
    const q = 3 # Number of power iterations in the RSVD
    const K = 120 # Set the rank of the RSVD for Q, take i.e. K =
                  # ceil(length(strue)/7) 50413416850

    Z = PCGA.randSVDzetas(Q,K,p,q) 
    numparams = length(strue) 
else
    println("example not supported")
end

Zis = Array{Float64, 1}[Z[:,1],Z[:,2]];

for i = 3:size(Z,2)
    Zis = push!(Zis,Z[:,i])
end

if CASEFLAG == 1
    mean_s = zeros(length(strue));
    #choose a random smooth field in the prior to start at
    U,S = svd(Q) #assuming Q not perfectly spd
    Sh = sqrt(S)
    L = U*diagm(Sh)
    srand(1)
    s0 = mean_s + 0.5* L * randn(length(strue));
elseif CASEFLAG == 2
    mean_s = 0.3*ones(length(strue));
    s0 =  0.6*ones(length(strue));
elseif CASEFLAG == 3
    mean_s = 0.3*ones(length(strue));
    #choose a random smooth field in the prior to start at
    U,S = svd(Q) #assuming Q not perfectly spd
    Sh = sqrt(S)
    L = U*diagm(Sh)
    srand(1)
    s0 = mean_s + 0.5* L * randn(length(strue));
elseif CASEFLAG == 4
    srand(1)
    mean_s = strue + 0.5 * randn(length(strue))
    s0 = 0.3*ones(length(strue));
elseif CASEFLAG == 5
    mean_s = zeros(length(strue));
    s0 = mean_s;
elseif CASEFLAG == 6
    mean_s = Array(Float64, length(strue))
    for i = 1:length(strue)
	w, krigingerror = Krigapping.krige(collect(coords[i]), xy_obs, cov)
	mean_s[i] = dot(w, k_obsNoise)
    end
    s0 = mean_s
else
    println("check mean and s0")
end

relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

tic()

for k = 1:total_iter
    sbar[:,k+1] = PCGA.pcgaiteration(testForward, sbar[:,k], mean_s, Zis, Gamma, yvec)
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end

totaltime_PCGA = toq() 

rank_QK = rank(Z*Z')
relerr_s_endminus1 = relerror[end-1]
relerr_s_end = relerror[end]
rounderr =  round(relerr_s_end*10000)/10000

@show(total_iter,relerr_s_endminus1, relerr_s_end, totaltime_PCGA, rank_QK,p,q,covdenom)

# Plotting for each example
if EXAMPLEFLAG == 1
    x = linspace(0,1,numparams);
   
    plot(x,strue,x,mean_s,x,sbar[:,1],x,sbar[:,end],linestyle="-",marker="o")
    legend(["sythetic","s_mean","initial s_0 (a random field in the
    prior probability distribution)","s_$(total_iter)"], loc=0)


    xlabel("unit 1D domain x")
    ylabel("1D parameter field s(x)")

    title("PCGA, total iterates=$total_iter, noise=$noise%,
    rank=$(rank_QK), p=$(p), q=$(q), relerr=$(rounderr)")
    grid("on")

    figure(2)
    plot(1:total_iter+1,relerror,linestyle="-",marker="o")
    title("Relative error vs iteration number, PCGA method")

elseif EXAMPLEFLAG == 2
    totfignum  = 5 

    k1mean, k2mean = x2k(mean_s) #mean_s is all 0's for case 1, 0.3 for
    #case 2
    logk_mean = ks2k(k1mean,k2mean)

    k1s0,k2s0 = x2k(s0)
    logk_s0 = ks2k(k1s0,k2s0)

    k1p_i,k2p_i = x2k(sbar[:,end-1]);
    logkp_i = ks2k(k1p_i,k2p_i);

    k1p,k2p = x2k(sbar[:,end]);
    logkp = ks2k(k1p,k2p);
    
    fig = figure(figsize=(6*totfignum, 6)) 
    
    vmin = minimum(logk)
    vmax = maximum(logk)

    plotfield(logk,totfignum,1,vmin,vmax)
    title("the true logk")

    plotfield(logk_mean,totfignum,2,vmin,vmax)
    plt.title("the mean, here truelogk + noise")

    plotfield(logk_s0,totfignum,3,vmin,vmax)
    plt.title("s0 (using prior and mean)")
    
    plotfield(logkp_i,totfignum,totfignum-1,vmin,vmax)
    plt.title("s_$(total_iter-1)")

    plotfield(logkp,totfignum,totfignum,vmin,vmax)
    plt.title("the last iterate, total_iter = $total_iter")

    ax1 = axes([0.92,0.1,0.01,0.8])   
    colorbar(cax = ax1)

    suptitle("2D example", fontsize=16)        

else
    println("example not supported")
end


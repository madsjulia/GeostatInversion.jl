#module PCGA
#Definetly not array optimized
using Debug
import PCGA


const numparams = 30

#close("all")

# Which example to run PCGA on
# 1 = deconvolution test problem
# 2 = ellen.jl forward model
const EXAMPLEFLAG = 2 

# Inputs into pcgaiteration 
# forwardmodel - param to obs map
#            s - current iterate          
#            X - mean of parameter prior (replace with B*X drift matrix
# later for p>1)
#          xis - K columns of Z where Q approx= ZZ^T
#            R - covariance of measurement error (data misfit term)
#            y - data vector            
tic()

if EXAMPLEFLAG == 1
    using PyPlot
    include("deconvolutionTestProblem.jl")
    G,strue,yvec,Gamma,C = deconv2(numparams,noise);
    Z = randSVDzetas(C); #Do random SVD on the prior part covariance matrix
elseif EXAMPLEFLAG == 2
    #     this_dir = dirname(@__FILE__);
    # include(abspath(joinpath(this_dir,
    # "../finitedifference2d.jl/ellen.jl")))
    include("ellen.jl")
    testForward = forwardObsPoints
    Gamma = R
    strue = [truelogk1[1:end]; truelogk2[1:end]] #vectorized 2D
    #parameter field
    yvec = u_obsNoise #see ellen.jl for noise level
    Z = PCGA.randSVDzetas(Q) 
else
    println("example not supported")
end


Zis = Array{Float64, 1}[Z[:,1],Z[:,2]];

for i = 3:size(Z,2)
    Zis = push!(Zis,Z[:,i])
end


#Runs the optimization loop until it converges
const total_iter = 2;
#s0 = strue+0.05*randn(length(strue));
#s0 = 0.5*ones(length(strue));
# s0 = zeros(length(strue));
# mean = zeros(length(strue));
# s0 = ones(length(strue));
# mean = ones(length(strue));


mean = strue + randn(length(strue));
#choose a random smooth field in the prior to start at
U,S = svd(Q) #assuming Q not perfectly spd
Sh = sqrt(S)
L = U*diagm(Sh)
srand(1)
s0 = mean +  L * randn(length(strue));


relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

for k = 1:total_iter
    #tic() 
   #sbar[:,k+1] = pcgaiteration(testForward, sbar[:,k], strue, Zis,
    #Gamma, yvec)
    sbar[:,k+1] = PCGA.pcgaiteration(testForward, sbar[:,k], mean, Zis, Gamma, yvec)
    #toc()
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end

return sbar,relerror

# this doesn't work need to check formation of etas
#  norm(HQHpR-(A'*C*A+Gamma))
# 2.059780712550066e18

toc()
rel_errPCGA = norm(sbar[:,end]-strue)/norm(strue);
@show(rel_errPCGA)

if EXAMPLEFLAG == 1
    x = linspace(0,1,numparams);
    plot(x,strue,x,sbar[:,1],x,sbar[:,end-2],x,sbar[:,end-1],x,sbar[:,end],linestyle="-",marker="o")
    legend(["sythetic","initial s_0","s_end-2","s_end-1","s_end"], loc=0)
    xlabel("unit 1D domain x")
    ylabel("1D parameter field s(x)")
    title("PCGA, total iterates = $total_iter, noise = $noise%")
    grid("on")

    figure(2)
    plot(1:total_iter+1,relerror,linestyle="-",marker="o")
    title("Relative error vs iteration number, PCGA method")

elseif EXAMPLEFLAG == 2
    fignum  = 5    

    k1mean, k2mean = x2k(mean)
    logk_mean = ks2k(k1mean,k2mean)

    k1s0,k2s0 = x2k(s0)
    logk_s0 = ks2k(k1s0,k2s0)

    k1p_i,k2p_i = x2k(sbar[:,end-1]);
    logkp_i = ks2k(k1p_i,k2p_i);

    k1p,k2p = x2k(sbar[:,end]);
    logkp = ks2k(k1p,k2p);
    
    fig = plt.figure(figsize=(6*fignum, 6))    
    
    plotfield(logk,1,fignum)
    plt.title("the true logk")

    plotfield(logk_mean,2,fignum)
    plt.title("the mean, here truelogk + noise")

    plotfield(logk_s0,3,fignum)
    plt.title("s0 (using prior and mean)")

    plotfield(logkp_i,fignum-1,fignum)
    plt.title("s_end-1")

    plotfield(logkp,fignum,fignum)
    plt.title("the last iterate, total_iter = $total_iter")

    vmin = minimum(logk)
    vmax = maximum(logk)
    plt.clim(vmin, vmax)
    #plt.colorbar() #this makes the resizing weird
    plt.suptitle("2D example", fontsize=16)        

    plt.show()
    
  
else
    println("example not supported")
end


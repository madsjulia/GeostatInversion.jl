import PCGA
import Krigapping
using JLD

const EXAMPLEFLAG = 2 
const CASEFLAG = 5
const RANDFLAG = 0

const SAVEFLAG = 0  #switch to 1 to save data

const LMFLAG = 1 #switch to false for vanilla PCGA, 1 means LM algo for GA
#in Nowak and Cirpka 2004

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
# RANDFLAG = 1   for reduction of PCGA system using Gauss. sketch. matrix

# Last updated August 6, 2015 by Ellen Le
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
const total_iter = 10
const pdrift = 1 #dimension of the drift matrix

if EXAMPLEFLAG == 1
    using PyPlot
    include("deconvolutionTestProblem.jl")
    const noise = 1  #  noise = 5 means 5% of max value
    const numparams = 30
    G,strue,yvec,Gamma,Q = deconv2(numparams,noise);

    const p = 0 # The oversampling parameter for increasing RSVD accuracy
    const q = 0 # Number of power iterations in the RSVD
    const K = 10 # Set the rank of the RSVD for Q, take i.e. K =
        # ceil(length(strue)/7) 

    Z = PCGA.randSVDzetas(Q,K,p,q); # Random SVD on the prior part covariance matrix
elseif EXAMPLEFLAG == 2
    include("ellen.jl") #get R, Q
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


if RANDFLAG == 1
    N  = size(R,2)
    Kred = 2000
    @show(Kred)
    srand(1)
    S_type = [1/sqrt(N)*randn(Kred,N) zeros(Kred,pdrift);zeros(pdrift,N) eye(pdrift,pdrift)];
    typeS = "Gaussian"
elseif RANDFLAG == 2
    #put Achlioptas, rad here
end

tic()

if LMFLAG == 1
    sbar,RMSE,cost,iterCt =  PCGA.pcgaiterationlm(testForward,s0,mean_s,Zis,Gamma,yvec,strue,
      maxIter=total_iter,lmoption=LMFLAG)
elseif RANDFLAG == 0
    sbar,RMSE,cost,iterCt =  PCGA.pcgaiteration(testForward,s0,mean_s,Zis,Gamma,yvec,strue,
                                                maxIter=total_iter)
elseif RANDFLAG == 1
    sbar,RMSE,cost,iterCt =  PCGA.pcgaiteration(testForward,s0,mean_s,Zis,Gamma,yvec, strue,
                                            maxIter=total_iter,randls=true,S=S_type)
else
    println("check LMFLAG,RANDFLAG")
end

totaltime_PCGA = toq() 

rank_QK = rank(Z*Z')
rmse_s_endminus1 = RMSE[iterCt-1]
rmse_s_end = RMSE[iterCt]
rounderr =  round(rmse_s_end*10000)/10000

# Plotting for each example
if EXAMPLEFLAG == 1
    x = linspace(0,1,numparams);
    
    figure()
    plot(x,strue,x,mean_s,x,sbar[:,1],x,sbar[:,end],linestyle="-",marker="o")
    legend(["sythetic","s_mean","initial s_0 (a random field in the
            prior probability distribution)","s_$(iterCt)"], loc=0)


    xlabel("unit 1D domain x")
    ylabel("1D parameter field s(x)")

    if RANDFLAG == 0

        title("PCGA, total iterates=$(iterCt), noise=$(noise)%,
              RMSE=$(rounderr), time=$(totaltime_PCGA)")

    else

        title("Randomized PCGA, total iterates=$(iterCt), noise=$(noise)%,
              RMSE=$(rounderr), time=$(totaltime_PCGA)")
    end

    grid("on")

    figure()
    plot(1:iterCt+1,relerror,linestyle="-",marker="o")
    title("Relative error vs iteration number, PCGA method")

elseif EXAMPLEFLAG == 2
    nrow = 2
    ncol = 4 

    k1mean, k2mean = x2k(mean_s) #mean_s is all 0's for case 1, 0.3 for
    #case 2
    logk_mean = ks2k(k1mean,k2mean)

    k1s0,k2s0 = x2k(s0)
    logk_s0 = ks2k(k1s0,k2s0)
    
    fig = figure(figsize=(6*ncol, 6*nrow)) 
    
    vmin = minimum(logk)
    vmax = maximum(logk)

    plotfield(logk,nrow,ncol,1,vmin,vmax)
    grid(linewidth=3)    
    title("the true logk")

    plotfield(logk_mean,nrow,ncol,2,vmin,vmax)
    grid(linewidth=3)    
    plt.title("the mean")

    plotfield(logk_s0,nrow, ncol,3,vmin,vmax)
    grid(linewidth=3)
    plt.title("s0")
    
    #plotting the iterates
    j=1
    for i = [1:4,10]
        k1p_i,k2p_i = x2k(sbar[:,i+1]);
        logkp_i = ks2k(k1p_i,k2p_i)
        plotfield(logkp_i,nrow,ncol,3+j,vmin,vmax)
        grid(linewidth=3)       
        plt.title("s_$(i)")
        j=j+1
    end

    if RANDFLAG == 0

        suptitle("PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom),
                 alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)",fontsize=16)        

    else

        astr = typeS*" Randomized PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom), alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)"
        suptitle(astr,fontsize=16)        
    end

    ax1 = axes([0.92,0.1,0.02,0.8])   
    colorbar(cax = ax1)
   

    figure()
    plot(0:iterCt,RMSE[1:iterCt+1],linestyle="-",marker="o")
    title("2D RMSE vs iteration number, PCGA method,
          noise=$(noise)%")'
    
    figure()
    plot(1:iterCt,cost[1:iterCt],linestyle="-",marker="o")
    title("2D cost vs iteration number, PCGA method,
          noise=$(noise)%"')

    k1p,k2p = x2k(sbar[:,end]);
    logkp = ks2k(k1p,k2p)

    if SAVEFLAG == 1
        if RANDFLAG == 0
            str="PCGAnoise$(noise)__al$(alpha)_cov$(covdenom)obs$(numobs).jld"
        elseif RANDFLAG == 1
            str="GRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)obs$(numobs)K$(Kred).jld"
        end
        @show(str)   
        println("saving")
        save(str,"sbar",sbar,"cost",cost,"totaltime_PCGA",totaltime_PCGA,"RMSE",RMSE,"iterCt",iterCt)
    elseif SAVEFLAG == 0
        println("not saving")
    end


else
    println("example not supported")
end


@show(iterCt,rmse_s_endminus1, rmse_s_end, totaltime_PCGA,
      rank_QK,p)

if EXAMPLEFLAG == 2
    @show(covdenom,alpha)
end

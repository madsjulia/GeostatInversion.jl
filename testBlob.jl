# Runs blobV2.jl

 # Last updated 20 August 2015 by Ellen Le
 # Questions: omalled@lanl.gov, ellenble@gmail.com

import PCGA
import Krigapping
using JLD

const RANDFLAG = 1
const SAVEFLAG = 0  #switch to 1 to save data

#Run the optimization loop until it converges or a maximum total_iter number of times
const total_iter = 10
const pdrift = 1 #dimension of the drift matrix

include("blobV2.jl") #get u_obs, truth log k, forward map, fcns to make Q,R

# for alpha = [1,5,10]
#     for covdenom = [0.05,0.1,0.2,0.3]
for alpha = [4,5,6]
    for covdenom = [0.025,0.05,0.075]




testForward = forwardObsPoints
strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field
Q = makeCovQ(strue,covdenom,alpha)

const noise = 5.0
yvec, Gamma = make_yandR(u_obs,noise)

const p = 20 # The oversampling parameter for increasing RSVD accuracy
const q = 3 # Number of power iterations in the RSVD
const K = 120 # Set the rank of the RSVD for Q, take i.e. K =
    # ceil(length(strue)/7) 50413416850

Z = PCGA.randSVDzetas(Q,K,p,q) 
numparams = length(strue) 

Zis = Array{Float64, 1}[Z[:,1],Z[:,2]];

for i = 3:size(Z,2)
    Zis = push!(Zis,Z[:,i])
end

mean_s = zeros(length(strue));
s0 = mean_s;

if RANDFLAG == 1
    N  = length(yvec)
    Kred = 2000
    #Kred = 250
    #Kred = 10
    @show(Kred)
    srand(1)
    S_type = 1/sqrt(N)*randn(Kred,N)
    typeS = "Gaussian"
elseif RANDFLAG == 2
    #put Achlioptas, rad here
end

if RANDFLAG == 0
    totaltime_PCGA = @elapsed sbar,RMSE,cost,iterCt =  PCGA.pcgaiteration(testForward,s0,mean_s,Zis,Gamma,yvec,strue,
                                                                          maxIter=total_iter)

elseif RANDFLAG == 1
    totaltime_PCGA = @elapsed sbar,RMSE,cost,iterCt =  PCGA.rgaiteration(testForward,s0,mean_s,Zis,Gamma,yvec, strue, S_type,
                                                                         maxIter=total_iter,randls=true)
end

#=
    rank_QK = rank(Z*Z')
    rmse_s_endminus1 = RMSE[iterCt-1]
    =#

rmse_s_end = RMSE[iterCt]

function plotresults(colorchoice)
    nrow = 2
    ncol = 4 

    k1mean, k2mean = x2k(mean_s) #mean_s is all 0's for case 1, 0.3 for
    #case 2
    logk_mean = ks2k(k1mean,k2mean)

    k1s0,k2s0 = x2k(s0)
    logk_s0 = ks2k(k1s0,k2s0)

    fig = figure(figsize=(3*ncol, 3*nrow)) 
#    fig = figure(figsize=(6*ncol, 6*nrow)) 

    # vmin = minimum(logk)
    # vmax = maximum(logk)

    vmin = -0.8
    vmax = 0.8

    plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true,mycmap=colorchoice)
    grid(linewidth=3)    
    title("the true logk")

    plotfield(logk_mean,nrow,ncol,2,vmin,vmax,noObs=true,mycmap=colorchoice)
    grid(linewidth=3)    
    plt.title("the mean")

    plotfield(logk_s0,nrow, ncol,3,vmin,vmax,noObs=true,mycmap=colorchoice)
    grid(linewidth=3)
    plt.title("s0")

    #plotting the iterates
    j=1
    for i = [1:4,iterCt]
        k1p_i,k2p_i = x2k(sbar[:,i+1]);
        logkp_i = ks2k(k1p_i,k2p_i)
        plotfield(logkp_i,nrow,ncol,3+j,vmin,vmax,noObs=true,mycmap=colorchoice)
        grid(linewidth=3)       
        plt.title("s_$(i)")
        j=j+1
    end

    rounderr =  round(rmse_s_end*10000)/10000
    if RANDFLAG == 0
        suptitle("PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom),
                 alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)",fontsize=16)        
    else
        astr = typeS*" Randomized PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom), alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)"
        suptitle(astr,fontsize=16)        
    end
    ax1 = axes([0.92,0.1,0.02,0.8])   
    colorbar(cax = ax1)
end

plotresults("seismic_r")

# figure()
# plot(0:iterCt,RMSE[1:iterCt+1],linestyle="-",marker="o")
# title("2D RMSE vs iteration number, PCGA method, noise=$(noise)%")

# figure()
# plot(1:iterCt,cost[1:iterCt],linestyle="-",marker="o")
# title("2D cost vs iteration number, PCGA method,
#       noise=$(noise)%")

if SAVEFLAG == 1
    if RANDFLAG == 0
        str="blobPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)obs$(numobs).jld"
    elseif RANDFLAG == 1
        str="blobGRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)obs$(numobs)K$(Kred).jld"
    end
    @show(str)   
    println("saving")
    save(str,"sbar",sbar,"cost",cost,"totaltime_PCGA",totaltime_PCGA,"RMSE",RMSE,"iterCt",iterCt)
elseif SAVEFLAG == 0
    println("not saving")
end

# @show(iterCt, rmse_s_end, totaltime_PCGA)
# @show(covdenom,alpha)

end
end


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

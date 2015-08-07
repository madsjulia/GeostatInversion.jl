#Compare the iterates of GRPCGA to PCGA
using JLD
using PyPlot

const noise = 5
const alpha = 20
const convdenom = 0.2
const Kred = 2000

include("ellen.jl")

nrow = 2
ncol = 4 


for RANDFLAG=[1,0]


    # vmin = minimum(logk)
    # vmax = maximum(logk)


    vmin = -0.8
    vmax = 0.8

    fig = figure(figsize=(4*ncol, 4*nrow))
    
    plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true)
    grid(linewidth=3)  

    title("the true logk")

    if RANDFLAG == 0
        str="PCGAnoise$(noise)__al$(alpha)_cov$(covdenom).jld"
        iterCt = 6
    elseif RANDFLAG == 1
        str="GRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)K$(Kred).jld"
        iterCt = 5
    end
    @show(str) 
    sbar = load(str,"sbar")
    totaltime_PCGA = load(str, "totaltime_PCGA")
    RMSE =  load(str, "RMSE")                     

    rmse_s_end = RMSE[iterCt]
    rounderr =  round(rmse_s_end*10000)/10000

    #plotting the differences
    j=1
    for i = [1:iterCt]
        k1p_i,k2p_i = x2k(sbar[:,i+1]);
        logkp_i = ks2k(k1p_i,k2p_i)
        plotfield(logkp_i-logk,nrow,ncol,1+j,vmin,vmax,noObs=true)
        grid(linewidth=3)
        RMSE_field = norm(logk[:]-logkp_i[:])*(1/sqrt(length(logk)))
        roundrmse = round(RMSE_field*10000)/10000
        @show(i,RMSE_field)

        plt.title("s_$(i)-s_truth,RMSE_field=$(roundrmse)")
        j=j+1
    end

    if RANDFLAG == 0

        suptitle("PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom),
                 alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)",fontsize=16)        

    else

        astr = "Gaussian Randomized PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom), alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)"
        suptitle(astr,fontsize=16)        
    end


    ax1 = axes([0.92,0.1,0.02,0.8])   
    colorbar(cax = ax1)

    #plotting the iterates again





    fig = figure(figsize=(4*ncol, 4*nrow)) 

    plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true)
    grid(linewidth=3)
    title("the true logk")

    if RANDFLAG == 0
        str="PCGAnoise$(noise)__al$(alpha)_cov$(covdenom).jld"
        iterCt = 6
    elseif RANDFLAG == 1
        str="GRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)K$(Kred).jld"
        iterCt = 5
    end
    @show(str) 
    sbar = load(str,"sbar")
    totaltime_PCGA = load(str, "totaltime_PCGA")
    RMSE =  load(str, "RMSE")                     

    rmse_s_end = RMSE[iterCt]
    rounderr =  round(rmse_s_end*10000)/10000

    j=1
    for i = [1:iterCt]
        k1p_i,k2p_i = x2k(sbar[:,i+1]);
        logkp_i = ks2k(k1p_i,k2p_i)
        plotfield(logkp_i,nrow,ncol,1+j,vmin,vmax,noObs=true)
        grid(linewidth=3)
        RMSE_field = norm(logk[:]-logkp_i[:])*(1/sqrt(length(logk)))
        roundrmse = round(RMSE_field*10000)/10000
        @show(i,RMSE_field)

        plt.title("s_$(i),RMSE_field=$(roundrmse)")
        j=j+1
    end

    if RANDFLAG == 0

        suptitle("PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom),
                 alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)",fontsize=16)        

    else

        astr = "Gaussian Randomized PCGA 2D, noise=$(noise)%,its=$(iterCt),covdenom=$(covdenom), alpha=$(alpha),RMSE=$(rounderr),time=$(totaltime_PCGA)"
        suptitle(astr,fontsize=16)        
    end


    ax1 = axes([0.92,0.1,0.02,0.8])   
    colorbar(cax = ax1)




    sbar = 0
    totaltime_PCGA = 0
    rounderr = 0
    rmse_s_end = 0
    RMSE = 0
end

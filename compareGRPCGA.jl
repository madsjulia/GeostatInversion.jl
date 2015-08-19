#Compare the iterates of GRPCGA to PCGA
using JLD
using PyPlot

const noise = 5
const alpha = 20
const convdenom = 0.2
const Kred = 2000

const PLOTFLAG = 4 #switch to 0 for all results, 1 for presentation
#results, 2 for newest presentation results, 3 for paper figures
#4 for scatter plots
include("ellen.jl")
strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field
#more plotting for presentation:


if PLOTFLAG == 4 
    
    close("all")

    # Plot 1: PCGA
    str="PCGAnoise$(noise)__al$(alpha)_cov$(covdenom).jld"
    @show(str)
    iterCt = 6
    sbar = load(str,"sbar")
    RMSE_PCGA = load(str,"RMSE")
    time_PCGA = load(str,"totaltime_PCGA")
    pearsonPCGA = cor(strue,sbar[:,iterCt])
    @show(RMSE_PCGA[iterCt],time_PCGA,iterCt,pearsonPCGA) 

    ## For the interpolated points:
    # k1p_i,k2p_i = x2k(sbar[:,iterCt]);
    # logkp_i = ks2k(k1p_i,k2p_i)
    # pearsonPCGA = cor(logk,logkp_i)
    # @show(RMSE_PCGA[iterCt],time_PCGA,iterCt,pearsonPCGA) 

    figure(figsize=(11.5,7.5))
    # scatter(logk, logkp_i, s=70, marker="+",facecolor="none",edgecolor="k",label="PCGA")
    scatter(strue, sbar[:,iterCt], s=70, marker="+",facecolor="none",edgecolor="k",label="PCGA")

    xmin,xmax = xlim()
    ymin,ymax = ylim()
    f(x) = x
    x = linspace(xmin,xmax,100)

    plot(x,f(x),linestyle="-",color="r",linewidth=3)
    vlines(0, ymin, ymax, color="r", linestyles="dashed",linewidth=3)

    axis([-2,1.5,-2,1.5])
    legend()

    xlabel("truth log k")
    ylabel("estimate")    
    title("PCGA Pearson cor = $(pearsonPCGA)")

    grid()
    savefig("../../paper/rpcgapaper/figures/scatpcga.pdf")

    #RGA comparison
    ##Plot 2: first RGA on top of PCGA
    figure(figsize=(11.5,7.5))
    # scatter(logk, logkp_i, s=70, marker="+",facecolor="none",edgecolor="k",label="PCGA")
    scatter(strue, sbar[:,iterCt], s=70, marker="+",facecolor="none",edgecolor="k",label="PCGA")

    xlabel("truth log k")
    ylabel("estimate")    

    rstr="GRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)K$(Kred).jld"
    @show(rstr)
    iterCt = 5
    rsbar = load(rstr,"sbar")
    RMSE_RGA = load(rstr,"RMSE")
    time_RGA = load(rstr,"totaltime_PCGA")
    pearsonRGA = cor(strue,rsbar[:,iterCt])
    @show(RMSE_RGA[iterCt],time_RGA,iterCt,pearsonRGA)    

    # rk1p_i,rk2p_i = x2k(rsbar[:,iterCt]);
    # rlogkp_i = ks2k(rk1p_i,rk2p_i)
    # pearsonRGA = cor(logk,rlogkp_i)
    # @show(RMSE_RGA[iterCt],time_RGA,iterCt,pearsonRGA)

    #  scatter(logk, rlogkp_i, s=70, marker="o",facecolor="none",edgecolor="b",label="RGA")
    scatter(strue, rsbar[:,iterCt], s=70, marker="o",facecolor="none",edgecolor="b",label="RGA")

    plot(x,f(x),linestyle="-",color="r",linewidth=3)
    vlines(0, ymin, ymax, color="r", linestyles="dashed",linewidth=3)
    
    axis([-2,1.5,-2,1.5])

    xlabel("truth log k")
    ylabel("estimate")    
    title("PCGA Pearson cor = $(pearsonPCGA), RGA Pearson cor = $(pearsonRGA)")

    grid("on")

    legend()
    savefig("../../paper/rpcgapaper/figures/scatCompare.pdf")

    ##Plot 3: PCGA on top of RGA
    figure(figsize=(11.5,7.5))
    
    xlabel("truth log k")
    ylabel("estimate")    
    
    # scatter(logk, rlogkp_i, s=70, marker="o",facecolor="none",edgecolor="b",label="RGA")
    # scatter(logk, logkp_i, s=70, marker="+",facecolor="none",edgecolor="k",label="PCGA")
    scatter(strue, rsbar[:,iterCt], s=70, marker="o",facecolor="none",edgecolor="b",label="RGA")
    scatter(strue, sbar[:,iterCt], s=70, marker="+",facecolor="none",edgecolor="k",label="PCGA") 
    
    plot(x,f(x),linestyle="-",color="r",linewidth=3)
    vlines(0, ymin, ymax, color="r", linestyles="dashed",linewidth=3)
    
    axis([-2,1.5,-2,1.5])

    xlabel("truth log k")
    ylabel("estimate")    
    title("PCGA Pearson cor = $(pearsonPCGA), RGA Pearson cor = $(pearsonRGA)")

    grid("on")

    legend()
    savefig("../../paper/rpcgapaper/figures/scatCompare2.pdf")

    #Plot 4: RGA by itself

    figure(figsize=(11.5,7.5))

    # scatter(logk, rlogkp_i, s=70, marker="o",facecolor="none",edgecolor="b",label="RGA")
    scatter(strue, rsbar[:,iterCt], s=70, marker="o",facecolor="none",edgecolor="b",label="RGA")

    plot(x,f(x),linestyle="-",color="r",linewidth=3)
    vlines(0, ymin, ymax, color="r", linestyles="dashed",linewidth=3)
    
    xlabel("truth log k")
    ylabel("estimate")    
    title("RGA Pearson cor = $(pearsonRGA)")

    grid("on")

    legend()
    axis([-2,1.5,-2,1.5])
    savefig("../../paper/rpcgapaper/figures/scatRpcga.pdf")


elseif PLOTFLAG == 3 
    
    close("all")

    nrow = 1
    ncol = 1
    vmin = -0.8
    vmax = 0.8

    #truth ln k
    figure()
    plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true)
    grid(linewidth=3)  
    ax2 = axes([0.87,0.1,0.04,0.8])
    colorbar(cax = ax2)
    savefig("../../paper/rpcgapaper/figures/logk.pdf")


elseif PLOTFLAG == 2

    close("all")

    nrow = 1
    ncol = 1
    vmin = -0.8
    vmax = 0.8

    #truth ln k
    figure()
    plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true)
    grid(linewidth=3)  
    ax2 = axes([0.04,0.1,0.04,0.8])
    colorbar(cax = ax2)
    savefig("../../Presentations/wmsym2015/figures/truelogk.pdf")

    #LM
    m=50
    its = 10
    lmstr="$(m)logkp_its$(its)_al$(alpha)_cov$(covdenom).jld";
    @show(lmstr)    
    iterCt = 10
    lmlogkp = load(lmstr,"logkp")
    RMSE_LM =  load(lmstr,"RMSE_LM")
    timeLM = load(lmstr,"timeLM")
    @show(RMSE_LM,iterCt,timeLM)
    figure()
    plotfield(lmlogkp,nrow,ncol,1,vmin,vmax,noObs=true)
    grid(linewidth=3)  
    savefig("../../Presentations/wmsym2015/figures/LM.pdf")
    
    # PCGA

    str="PCGAnoise$(noise)__al$(alpha)_cov$(covdenom).jld"
    @show(str)
    iterCt = 6
    sbar = load(str,"sbar")
    RMSE_PCGA = load(str,"RMSE")
    time_PCGA = load(str,"totaltime_PCGA")
    @show(RMSE_PCGA[iterCt],time_PCGA,iterCt)
    k1p_i,k2p_i = x2k(sbar[:,iterCt]);
    logkp_i = ks2k(k1p_i,k2p_i)
    figure()
    plotfield(logkp_i,nrow,ncol,1,vmin,vmax,noObs=true)
    grid(linewidth=3)  
    savefig("../../Presentations/wmsym2015/figures/PCGA.pdf")
    
    #RGA
    rstr="GRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)K$(Kred).jld"
    @show(rstr)
    iterCt = 5
    rsbar = load(rstr,"sbar")
    RMSE_RGA = load(rstr,"RMSE")
    time_RGA = load(rstr,"totaltime_PCGA")
    @show(RMSE_RGA[iterCt],time_RGA,iterCt)
    rk1p_i,rk2p_i = x2k(rsbar[:,iterCt]);
    rlogkp_i = ks2k(rk1p_i,rk2p_i)
    figure()
    plotfield(rlogkp_i,nrow,ncol,1,vmin,vmax,noObs=true)      
    grid(linewidth=3)
    savefig("../../Presentations/wmsym2015/figures/RGA.pdf")


elseif PLOTFLAG == 1

close("all")
nrow = 1
ncol = 2
vmin = -0.8
vmax = 0.8

dvmin = -0.8
dvmax = 0.8
fig = figure(figsize=(4*3, 4*row)) #compare diffss in fig 1
fig2 = figure(figsize=(4*ncol, 4*nrow)) #compare pcga results in fig 2
plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true)
grid(linewidth=3)  

figure(1)
plotfield(logk,1,3,1,vmin,vmax,noObs=true) #compare diffs
grid(linewidth=3)  
ax2 = axes([0.04,0.1,0.02,0.8])
colorbar(cax = ax2)

str="PCGAnoise$(noise)__al$(alpha)_cov$(covdenom).jld"
iterCt = 6
sbar = load(str,"sbar")
k1p_i,k2p_i = x2k(sbar[:,iterCt]);
logkp_i = ks2k(k1p_i,k2p_i)

figure(2) #results
plotfield(logkp_i,nrow,ncol,2,vmin,vmax,noObs=true)
grid(linewidth=3)  

fig3 = figure(figsize=(4*ncol, 4*nrow)) #rpcga results in fig 3
plotfield(logk,nrow,ncol,1,vmin,vmax,noObs=true)
grid(linewidth=3)  


rstr="GRPCGAnoise$(noise)__al$(alpha)_cov$(covdenom)K$(Kred).jld"
iterCt = 5
rsbar = load(rstr,"sbar")
rk1p_i,rk2p_i = x2k(rsbar[:,iterCt]);

rlogkp_i = ks2k(rk1p_i,rk2p_i)

plotfield(rlogkp_i,nrow,ncol,2,vmin,vmax,noObs=true)

grid(linewidth=3)  
ax1 = axes([0.92,0.1,0.02,0.8])
colorbar(cax = ax1)
figure(1)
plotfield(logkp_i-logk,1,3,2,dvmin,dvmax,noObs=true,mycmap="seismic")
grid(linewidth=3)  
plotfield(rlogkp_i-logk,1,3,3,dvmin,dvmax,noObs=true,mycmap="seismic")
grid(linewidth=3)  

ax1 = axes([0.92,0.1,0.02,0.8])
colorbar(cax = ax1)



### plot LM results for comparison

close("all")
nrow = 1
ncol = 2
vmin = -0.8
vmax = 0.8

dvmin = -0.8
dvmax = 0.8


fig4 = figure(figsize=(4*2, 4*1)) #compare diffss in fig 1

plotfield(logk,1,2,1,vmin,vmax,noObs=true)
grid(linewidth=3)  

m=50
its = 10
lmstr="$(m)logkp_its$(its)_al$(alpha)_cov$(covdenom).jld";
iterCt = 10
lmlogkp = load(lmstr,"logkp")

plotfield(lmlogkp,1,2,2,vmin,vmax,noObs=true)
grid(linewidth=3)  
ax1 = axes([0.92,0.1,0.02,0.8])
colorbar(cax = ax1)

elseif PLOTFLAG == 0#plots from gitlab journal 
nrow = 2
ncol = 4 

for RANDFLAG=[1,0]
    
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
        plotfield(logkp_i-logk,nrow,ncol,1+j,vmin,vmax,noObs=true,mycmap="seismic")
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
end

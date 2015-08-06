using Calculus
using Optim
using JLD
#import YouzuoLM

# Test 2D example using Lev-Marq

# Last updated July 29, 2015 by Ellen Le
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

PLOTFLAG = 1
SAVEFLAG = 1

include("finite_difference.jl")
include("ellen.jl") #get R, Q

# save("diffAlphasCovs.jld", "logk", logk)

testForward = forwardObsPoints
strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field

yvec = u_obsNoise #see ellen.jl for noise level

L = chol(inv(R),:U)
S = chol(inv(Q),:U)

## check
# norm(L'*L-inv(C))
# norm(S'*S-inv(Gamma))

s0 = zeros(length(strue));
#mu = 0.3*ones(length(strue));
mu = s0

function f_lm(s::Vector)
    result = [L*(yvec - testForward(s)); S*(s-mu)]
    return result
end

tic()
function g_lm(s::Vector)
    J =  finite_difference_jacobian(f_lm,s,:forward)
#    J =  finite_difference_jacobian(f_lm,s)
return J
end
gradTime = toq()

initial_s = zeros(length(strue));

tic()

# Set trace to true to see iterates and call results.trace
results = Optim.levenberg_marquardt(f_lm, g_lm, initial_s, tolX=1e-15, tolG=1e-15, maxIter=13, lambda=200.0, show_trace=true)
#results, () = YouzuoLM.levenberg_marquardt(f_lm, g_lm, initial_s, tolX=1e-15, tolG=1e-15, maxIter=13, lambda=200.0, show_trace=true,solver_option=3)

timeLM = toq()


vmin = minimum(logk)
vmax = maximum(logk)

# norm(S'*S-inv(Gamma))
k1p,k2p = x2k(results.minimum);
logkp = ks2k(k1p,k2p);

if PLOTFLAG == 1
  
    errLM = norm(results.minimum-strue)/norm(strue)
    rounderr =  round(errLM*10000)/10000

    fig = figure(figsize=(6*2, 6)) 

    plotfield(logk,1,2,1,vmin,vmax)
    title("the true logk, grid size m=$(m)")

    plotfield(logkp,1,2,2,vmin,vmax)
    title("LM 2D,
          its=$(results.iterations),covdenom=$(covdenom),alpha=$(alpha),err=$(rounderr)")

    ax1 = axes([0.92,0.1,0.02,0.8])   
    colorbar(cax = ax1)


    figure()
    x = 1:(length(s0)+length(yvec))
    plot(x,abs(f_lm(strue)),x,abs(f_lm(s0)),x,abs(f_lm(results.minimum)),linestyle="-",marker="o")
    title("|f(s)|, LM 2D, its=$(results.iterations),covdenom=$(covdenom),alpha=$(alpha)")
    
    legend(["at s_true","at s0","at s_min"])

    figure()
    x2 = 1:(length(s0))
    plot(x2,abs(S*(strue-mu)),x2, abs(S*(s0-mu)), x2, abs(S*(results.minimum-mu)),linestyle="-",marker="o")
    title("|S*(s-mu)|, LM 2D, its=$(results.iterations),covdenom=$(covdenom),alpha=$(alpha)")
    legend(["at s_true","at s0","at s_min"]) 

    @show(errLM,timeLM, alpha,covdenom,results.iterations,gradTime)

else
    println("not plotting")
end

@show(timeLM,gradTime)

if SAVEFLAG == 1
    str="$(m)logkp_its$(results.iterations)_al$(alpha)_cov$(covdenom).jld"
    @show(str)
    save(str,"logkp",logkp,"timeLM",timeLM,"errLM",errLM)
else
    println("not saving min logK")
end

# Q[1:5,1:5]


# Plots all mins after all runs are saved separately
# include("ellen.jl")

# ncol = 4
# nrow = 2

# fig = figure(figsize=(6*ncol, 6*nrow)) 

# vmin = minimum(logk)
# vmax = maximum(logk)

# plotfield(logk,ncol,1,vmin,vmax)
# title("the true logk")

# i=2
# for alpha = [4,8,80,800,8000,80000,800000]
#     str="logkp_its$(results.iterations)_al$(alpha)_cov$(covdenom).jld"
#     logkp = load(str,"logkp")

#     plotfield(logkp,ncol,i,vmin,vmax)
#     title("LM 2D,
#           its=$(results.iterations),covdenom=$(covdenom),alpha=$(alpha)")
#     i = i+1
# end

# ax1 = axes([0.92,0.1,0.02,0.8])   
# colorbar(cax = ax1)




# #Plotting different covariances
# include("ellen.jl")
# ncol = 2
# nrow = 2

# fig = figure(figsize=(6*ncol, 6*nrow)) 

# vmin = minimum(logk)
# vmax = maximum(logk)

# plotfield(logk,ncol,1,vmin,vmax)
# title("the true logk")

# alpha = 800
# i=2
# for covdenom = [0.1,0.2,0.3]
#     str="logkp_its$(results.iterations)_al$(alpha)_cov$(covdenom).jld"
#     logkp = load(str,"logkp")

#     plotfield(logkp,ncol,i,vmin,vmax)
#     title("LM 2D,
#           its=$(results.iterations),covdenom=$(covdenom),alpha=$(alpha)")
#     i = i+1
# end


# ax1 = axes([0.92,0.1,0.02,0.8])   
# colorbar(cax = ax1)





# Plotting different iterations
# ncol = 4
# nrow = 2

# fig = figure(figsize=(6*ncol, 6*nrow)) 

# vmin = minimum(logk)
# vmax = maximum(logk)

# plotfield(logk,ncol,1,vmin,vmax)
# title("the true logk")

# # Plotting different iterations
# i=2
# for its = [1,5,10,13,100,10000]                     
#     str="logkp_its$(its)_al$(alpha)_cov$(covdenom).jld"
#     logkp = load(str,"logkp")
#     plotfield(logkp,ncol,i,vmin,vmax)
#     title("LM 2D, its=$(its),covdenom=$(covdenom),alpha=$(alpha)")
#     i = i+1
# end

# Tweaking the 50 by 50 case
nrow = 3
ncol = 2
m=50

its = 13

fig = figure(figsize=(6*ncol, 6*nrow)) 

vmin = minimum(logk)
vmax = maximum(logk)

plotfield(logk,nrow,ncol,5,vmin,vmax)
title("the true logk")

i=1
for covdenom = [0.2,0.3]
    for alpha =[800,8000]
        str="$(m)logkp_its$(its)_al$(alpha)_cov$(covdenom).jld"
        logkp = load(str,"logkp")
        plotfield(logkp,nrow,ncol,i,vmin,vmax)
        i = i+1
        errLM = load(str,"errLM") 
        rounderr =  round(errLM*10000)/10000
        title("LM 2D, its=$(its),covdenom=$(covdenom),
alpha=$(alpha),err=$(rounderr)",fontsize=16)        
    end
end

suptitle("LM 2D",fontsize=16)        

ax1 = axes([0.92,0.1,0.02,0.8])   
colorbar(cax = ax1)

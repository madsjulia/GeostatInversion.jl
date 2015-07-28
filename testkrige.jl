import PCGA
import Krigapping
using Debug

include("ellen.jl")
testForward = forwardObsPoints
Gamma = R
strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field
yvec = u_obsNoise # see ellen.jl for noise level

const p = 20 # The oversampling parameter for increasing RSVD accuracy
const q = 3 # Number of power iterations in the RSVD
const K = 120 # Set the rank of the RSVD for Q, take i.e. K =
    # ceil(length(strue)/7) 

Z = PCGA.randSVDzetas(Q,K,p,q) 

Zis = Array{Float64, 1}[Z[:,1],Z[:,2]];

for i = 3:size(Z,2)
    Zis = push!(Zis,Z[:,i])
end

#Runs the optimization loop until it converges
const total_iter = 7;

#mean = strue + randn(length(strue));
mean = Array(Float64, length(strue))
for i = 1:length(strue)
	w, krigingerror = Krigapping.krige(collect(coords[i]), xy_obs, cov)
	mean[i] = dot(w, k_obsNoise)
end

s0 = mean

relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

tic()

for k = 1:total_iter
   #sbar[:,k+1] = pcgaiteration(testForward, sbar[:,k], strue, Zis,
    #Gamma, yvec)
    sbar[:,k+1] = PCGA.pcgaiteration(testForward, sbar[:,k], mean, Zis, Gamma, yvec)
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end

totaltime_PCGA = toq() 

rank_QK = rank(Z*Z')
rel_errPCGA = norm(sbar[:,end]-strue)/norm(strue);
@show(total_iter,rel_errPCGA, totaltime_PCGA, rank_QK,p,q,covdenom)

totfignum  = 5 

k1mean, k2mean = x2k(mean) #mean_s is all 0's for case 1, 0.3 for
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

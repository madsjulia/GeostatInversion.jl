import PCGA
import Krigapping
using Debug

include("ellen.jl")
testForward = forwardObsPoints
Gamma = R
strue = [truelogk1[1:end]; truelogk2[1:end]] #vectorized 2D
#parameter field
yvec = u_obsNoise #see ellen.jl for noise level
Z = PCGA.randSVDzetas(Q) 
Zis = Array{Float64, 1}[Z[:,1],Z[:,2]];
for i = 3:size(Z,2)
    Zis = push!(Zis,Z[:,i])
end

#Runs the optimization loop until it converges
const total_iter = 10;

#mean = strue + randn(length(strue));
mean = Array(Float64, length(strue))
for i = 1:length(strue)
	w, krigingerror = Krigapping.krige(collect(coords[i]), xy_obs, cov)
	mean[i] = dot(w, k_obsNoise)
end
#choose a random smooth field in the prior to start at
U,S = svd(Q) #assuming Q not perfectly spd
Sh = sqrt(S)
L = U*diagm(Sh)
srand(1)
#s0 = zeros(size(mean))
s0 = mean

relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

for k = 1:total_iter
   #sbar[:,k+1] = pcgaiteration(testForward, sbar[:,k], strue, Zis,
    #Gamma, yvec)
    sbar[:,k+1] = PCGA.pcgaiteration(testForward, sbar[:,k], mean, Zis, Gamma, yvec)
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end

return sbar,relerror

rel_errPCGA = norm(sbar[:,end]-strue)/norm(strue);
@show(rel_errPCGA)

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
plt.clim(minimum(logk), maximum(logk))
plt.title("the true logk")

plotfield(logk_mean,2,fignum)
plt.clim(minimum(logk), maximum(logk))
plt.title("the mean, here truelogk + noise")

plotfield(logk_s0,3,fignum)
plt.clim(minimum(logk), maximum(logk))
plt.title("s0 (using prior and mean)")

plotfield(logkp_i,fignum-1,fignum)
plt.clim(minimum(logk), maximum(logk))
plt.title("s_end-1")

plotfield(logkp,fignum,fignum)
plt.clim(minimum(logk), maximum(logk))
plt.title("the last iterate, total_iter = $total_iter")

vmin = minimum(logk)
vmax = maximum(logk)
#plt.clim(vmin, vmax)
#plt.colorbar() #this makes the resizing weird
plt.suptitle("2D example", fontsize=16)        

plt.show()

using Calculus
using Optim

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


include("finite_difference.jl")
include("ellen.jl") #get R, Q
testForward = forwardObsPoints
strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field

yvec = u_obsNoise #see ellen.jl for noise level

L = chol(inv(R),:U)
S = chol(inv(Q),:U)

## check
# norm(L'*L-inv(C))
# norm(S'*S-inv(Gamma))

s0 = zeros(length(strue));
mu = 0.3*ones(length(strue));

function f_lm(s::Vector)
    result = [L*(yvec - testForward(s)); S*(s-mu)]
    return result
end


function g_lm(s::Vector)
   J =  finite_difference_jacobian(f_lm,s)
return J
end

initial_s = zeros(length(strue));
results = Optim.levenberg_marquardt(f_lm, g_lm, initial_s)


figure()


vmin = minimum(logk)
vmax = maximum(logk)

k1p,k2p = x2k(results.minimum);
logkp = ks2k(k1p,k2p);

imshow(transpose(logkp), extent=[c, d, a, b],interpolation="nearest")
clim(vmin,vmax)
colorbar()
for i = 1:numobs
    plot(observationpoints[1, i], observationpoints[2, i], ".", color="#E0B0FF")
end
title("Levenberg-Marqhart on 2D example, its=$(results.iterations)")


figure()
x = 1:(length(s0)+length(yvec))
plot(x,abs(f_lm(s0)),x,abs(f_lm(results.minimum)),linestyle="-",marker="o")
title("|f(s)| after Levenberg-Marqhart on 2D example, its=$(results.iterations)")
legend(["at s0","at s_min"])

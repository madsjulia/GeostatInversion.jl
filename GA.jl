using PyPlot

const EXAMPLEFLAG = 2 

# Runs tests for GA, the full Gauss-Newton method approximated by PCGA. 
# 2 examples available.
# Set:
# EXAMPLEFLAG = 1  for 1D deconvolution test problem
# EXAMPLEFLAG = 2  for 2D groundwater forward model

# Last updated July 17, 2015 by Ellen Le
# Questions: ellenble@gmail.com

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

const numparams = 30
const noise = 4

if EXAMPLEFLAG == 1
    include("deconvolutionTestProblem.jl")
    G,strue,yvec,Gamma,C = deconv2(numparams,noise);
	yvec = vec(yvec[:, 1])
	#yvec = testForward(strue)
elseif EXAMPLEFLAG == 2
    include("~/codes/finitedifference2d.jl/ellen.jl")
else
    println("example not supported")
end

function gaiteration(forwardmodel::Function,s::Vector, X::Vector, Q::Matrix, R::Matrix, y::Vector,H::Matrix)
    # Inputs: 
    # forwardmodel - param to obs map h(s)
    #            s - current iterate s_k or sbar          
    #            X - mean of parameter prior (replace with B*X drift matrix
    # later for p>1)
    #            Q - K-dim prior covariance matrix
    #            R - covariance of measurement error (data misfit term)
    #            y - data vector 
    #            H - linearized forward operator    

    global delta
    p = 1
    HQHpR = H'*Q*H + R
    HQ = H*Q
    HX = H*X
    Hs = H*s
    bigA = [HQHpR HX; transpose(HX) zeros(p, p)]
    b = [y - testForward(s) + Hs; zeros(p)]
    x = bigA \ b # we will replace this with a Krylov solver or something
    # like UMFPACK?
    s_new = X * x[end] + Q*H' * x[1:end-1]
    return s_new 
end

#Run the optimization loop until it converges or a total_iter number of times
const total_iter = 200;
s0 = zeros(length(strue));
relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

tic()
for k = 1:total_iter
    sbar[:,k+1] = gaiteration(testForward, sbar[:,k], strue, C, Gamma, yvec,G)
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end
time_GA = toq()

return sbar,relerror

x = linspace(0,1,numparams);
plot(x,strue,x,sbar[:,1],x,sbar[:,end-2],x,sbar[:,end-1],x,sbar[:,end],linestyle="-",marker="o")
legend(["sythetic","initial s_0","s_end-2","s_end-1","s_end"],loc= 0)
xlabel("unit 1D domain x")
ylabel("1D parameter field s(x)")
title("GA Method, total iterates = $total_iter, noise = $noise")
grid("on")

figure(2)
plot(1:total_iter+1,relerror,linestyle="-",marker="o")
title("Relative error vs iteration number, GA method")

relErrGA = norm(sbar[:,end]-strue)/norm(strue)
@show(relErrGA,time_GA,total_iter)

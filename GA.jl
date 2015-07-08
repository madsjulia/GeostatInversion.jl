#module GA
#Definetly not array optimized
#using Debug
using PyPlot

const numparams = 30

close("all")

const EXAMPLEFLAG = 1 # 1 = deconvolution test problem
# 2 = ellen.jl forward model

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


# @debug function pcgaiteration(s, X, xis, R, y; forwardmodel = testForward)
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
#	println(s_new - s)
    return s_new #, HQ,HQHpR
end


#Runs the optimization loop until it converges
const total_iter = 200;
#s0 = strue+0.05*randn(length(strue));
#s0 = 0.5*ones(length(strue));
s0 = zeros(length(strue));
relerror = Array(Float64,total_iter+1)
sbar  = Array(Float64,length(strue),total_iter+1)
sbar[:,1] = s0;
relerror[1] = norm(sbar[:,1]-strue)/norm(strue);

for k = 1:total_iter
    sbar[:,k+1] = gaiteration(testForward, sbar[:,k], strue, C, Gamma, yvec,G)
    relerror[k+1] = norm(sbar[:,k+1]-strue)/norm(strue);
end

return sbar,relerror

# this doesn't work need to check formation of etas
#  norm(HQHpR-(A'*C*A+Gamma))
# 2.059780712550066e18

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

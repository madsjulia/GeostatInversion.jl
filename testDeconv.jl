using Optim
using Calculus

# By running this code we show that the deconvolution problem (or the 2D
# problem) with zero
# mean and starting point converges via Newton's method in 3 iterations

const EXAMPLEFLAG = 1 
const alpha = 100
const maxit = 5
const tol = 1e-10
methods = ["l_bfgs", "momentum_gradient_descent","gradient_descent", "bfgs","cg","newton"]
#methods = ["cg"]

if EXAMPLEFLAG == 1
    using PyPlot
    const numparams=30
    include("deconvolutionTestProblem.jl")
    noise = 5  # what percent noise i.e. noise =5 means 5% of max value
    A,strue,yvec,Gamma,C = deconv2(numparams,noise);
elseif EXAMPLEFLAG == 2
    include("ellen.jl")
    testForward = forwardObsPoints
    Gamma = R
    strue = [truelogk1[:]; truelogk2[:]] #vectorized 2D parameter field
    yvec = u_obsNoise #see ellen.jl for noise level
    C = Q
else
    println("example not supported")
end

s0 = zeros(length(strue));

function cost(s::Vector)
    if EXAMPLEFLAG == 1
        c = 0.5.*(yvec-A*s)'*inv(Gamma)*(yvec-A*s) + alpha*0.5.*(s-s0)'*inv(C)*(s-s0)
        c = reshape(c,1)[1]
    elseif EXAMPLEFLAG == 2
        c = 0.5.*(yvec-testForward(s))'*inv(Gamma)*(yvec-testForward(s)) + 0.5.*(s-s0)'*inv(C)*(s-s0)
        c = reshape(c,1)[1]    
    else
        println("example not supported")
    end
    return c
end

#the exact gradient for 1D example, approx for 2D 
if EXAMPLEFLAG==1 
    function cost_gradient!(x::Vector, y::Vector)
        result = A'*((Gamma\A)*x) + C\x - A'*(Gamma\yvec)-(C\s0) 
	for i = 1:length(y)
	    y[i] = result[i]
	end
	return result
    end
elseif EXAMPLEFLAG==2
    function cost_gradient!(x::Vector, y::Vector)
	result = gradient(cost, x)
	for i = 1:length(y)
	    y[i] = result[i]
	end
	return result
    end
else
    println("no gradient available")
end

#approx hessian   
function cost_hessian!(x::Vector, y::Matrix)
	result = hessian(cost, x)
	for i = 1:length(y)
		y[i] = result[i]
	end
    return result
end

#f = cost
g! = cost_gradient!
h! = cost_hessian!

for method in methods #make a new graph for each method
meth = symbol(method)

res1 = optimize(cost, g!, h!, s0, method = meth, ftol = tol, grtol = tol, iterations = 1) 
s1 = res1.minimum;

tic()   
res2 = optimize(cost, g!, h!, s0, method = meth , ftol = tol, grtol =
                tol, iterations = maxit)
timeOpt = toq()
s_end = res2.minimum;

relErr_s1 =  norm(s1-strue)/norm(strue)
relErr_end = norm(s_end-strue)/norm(strue)

rounderr =  round(relErr_end*10000)/10000
its = res2.iterations

figure()

if EXAMPLEFLAG == 1

    x = linspace(0,1,numparams);   
    plot(x,strue,x,s0,linestyle="-",marker="o")
    plot(x,s1,linestyle="-",marker="o")
    plot(x,s_end,linestyle="-",marker="o")    
    title("$(res2.method), noise = $noise%, total iterates = $its, relerr=$(rounderr)")
    legend(["synthetic","s_0 = s_mean","s_1","s_$its"],loc= 0)
    grid("on")

elseif EXAMPLEFLAG == 2
    totfignum  = 3 

    k1s1,k2s1 = x2k(s1)
    logk_s1 = ks2k(k1s1,k2s1)

    k1_end,k2_end = x2k(s_end)
    logk_end = ks2k(k1_end,k2_end)
    
    fig = figure(figsize=(6*totfignum, 6)) 
    
    vmin = minimum(logk)
    vmax = maximum(logk)

    plotfield(logk,totfignum,1,vmin,vmax)
    title("the truth logk")

    plotfield(logk_s1,totfignum,2,vmin,vmax)
    title("interpolated logk after 1 iteration")

    plotfield(logk_end,totfignum,3,vmin,vmax)
    title("interpolated logk after $(its) iterations")
    
    ax1 = axes([0.92,0.1,0.01,0.8])   
    colorbar(cax = ax1)
    suptitle("$(res2.method), noise = $noise%, total iterates = $its, relerr=$(rounderr)", fontsize=16)        

else
    println("example not supported")
end

@show(res2.method,its,relErr_s1,relErr_end,timeOpt)

end

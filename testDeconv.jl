using Optim
using Calculus

# By running this code we show that the deconvolution problem (or the 2D
# problem) with zero
# mean and starting point converges via Newton's method in 3 iterations

const EXAMPLEFLAG = 1 
const alpha = 100
const maxit = 10
const tol = 1e-15
methods = ["l_bfgs", "momentum_gradient_descent","gradient_descent", "bfgs","cg","newton"]

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

f = cost
g! = cost_gradient!
h! = cost_hessian!

for method in methods #make a new graph for each method
meth = symbol(method)

res1 = optimize(f, g!, h!, s0, method = meth, ftol = tol, grtol = tol, iterations = 1) 
s1 = res1.minimum;

tic()   
res2 = optimize(f, g!, h!, s0, method = meth , ftol = tol, grtol = tol, iterations = maxit)
timeOpt = toq()
s_end = res2.minimum;

figure()

if EXAMPLEFLAG == 1
    x = linspace(0,1,numparams);   
    plot(x,strue,x,s0,linestyle="-",marker="o")
    plot(x,s1,linestyle="-",marker="o")
    plot(x,s_end,linestyle="-",marker="o")    
elseif EXAMPLEFLAG == 2

else
    println("example not supported")
end

grid("on")
its = res2.iterations
legend(["synthetic","s_0 = s_mean","s_1","s_$its"],loc= 0)

title("$(res2.method), total iterates = $its, noise = $noise%")

relErr = norm(s_end-strue)/norm(strue)
@show(res2.method,its,relErr,timeOpt)

end

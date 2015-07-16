# By running this code we show that the deconvolution problem (or the 2D
# problem) with zero
# mean and starting point converges via Newton's method in 3 iterations

using Optim
using Calculus

const numparams=30
#const noise = 4
const EXAMPLEFLAG = 1 
const alpha = 100

tic()

if EXAMPLEFLAG == 1
    using PyPlot
    #close("all")
    include("deconvolutionTestProblem.jl")
    noise = 5  # what percent noise i.e. noise =5 means 5% of max value
    A,strue,yvec,Gamma,C = deconv2(numparams,noise);
elseif EXAMPLEFLAG == 2
    include("ellen.jl")
    testForward = forwardObsPoints
    Gamma = R
    strue = [truelogk1[1:end]; truelogk2[1:end]] #vectorized 2D parameter field
    yvec = u_obsNoise #see ellen.jl for noise level
    C = Q
else
    println("example not supported")
end

#s0 = strue + randn(length(strue));

s0 = zeros(length(strue));

#choose a random smooth field around 0
U,S = svd(C) #assuming Q not perfectly spd
Sh = sqrt(S)
L = U*diagm(Sh)
srand(1)
s0 = s0 +  0.1*L * randn(length(strue));


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

# function cost_gradient!(x::Vector, y::Vector)
# 	result = gradient(cost, x)
# 	for i = 1:length(y)
# 		y[i] = result[i]
# 	end
# 	return result
# end

#the exact gradient below
function cost_gradient!(x::Vector, y::Vector)
    result = A'*((Gamma\A)*x) + C\x - A'*(Gamma\yvec)-(C\s0) 
	for i = 1:length(y)
	      y[i] = result[i]
	end
	return result
end

function cost_hessian!(x::Vector, y::Matrix)
	result = hessian(cost, x)
	for i = 1:length(y)
		y[i] = result[i]
	end
	return result
end


if EXAMPLEFLAG == 1

    # # Nelder mead:
    # res = optimize(cost, s0, iterations = 20,store_trace = true,)

    # x = linspace(0,1,numparams)
    # figure()
    # plot(x,strue,x,s0,x,res.minimum,linestyle="-",marker="o")

    # res = optimize(cost, s0, iterations = 50,store_trace = true,)
    # plot(x,res.minimum,linestyle="-",marker="o")

    # res = optimize(cost, s0, iterations = 100,store_trace = true,)
    # plot(x,res.minimum,linestyle="-",marker="o")

    # res = optimize(cost, s0, store_trace = true,)
    # plot(x,res.minimum,linestyle="-",marker="o")

    # n = res.iterations;

    # legend(["synthetic","initial","s_20","s_50","s_100","min = s_$n"])


    # Newton:
    f = cost
    g! = cost_gradient!
    h! = cost_hessian!

    figure()
    tol = 1e-15
    meth =: l_bfgs #newton #cg, l_bfgs, bfgs, gradient_descent,
    #momentum_gradient descent
    res = optimize(f, g!, h!, s0, method = meth, ftol = tol, grtol = tol, iterations = 1) 
    s1 = res.minimum;
    x = linspace(0,1,numparams);   
    plot(x,strue,x,s0,linestyle="-",marker="o")
    plot(x,s1,linestyle="-",marker="o")

    maxit = 10

    res = optimize(f, g!, h!, s0, method = meth , ftol = tol, grtol = tol, iterations = maxit)

    toc()

    s_end = res.minimum;
    plot(x,s_end,linestyle="-",marker="o")
    grid("on")
    its = res.iterations
    legend(["synthetic","initial s_0","s_1","s_$its"],loc= 0)

    title("$(res.method), total iterates = $its, noise = $noise%")

    relErr = norm(s_end-strue)/norm(strue)
    @show(res.method)
    @show(relErr)
    

elseif EXAMPLEFLAG == 2




else
    println("example not supported")
end

#names(res)


# using Optim

# function rosenbrock(x::Vector)
#     return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
# end

# function rosenbrock_gradient!(x::Vector, storage::Vector)
#     storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
#     storage[2] = 200.0 * (x[2] - x[1]^2)
# end

# function rosenbrock_hessian!(x::Vector, storage::Matrix)
#     storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
#     storage[1, 2] = -400.0 * x[1]
#     storage[2, 1] = -400.0 * x[1]
#     storage[2, 2] = 200.0
# end


# f = rosenbrock
# g! = rosenbrock_gradient!
# h! = rosenbrock_hessian!

# x0 =  [0.0, 0.0]

# optimize(f,x0)

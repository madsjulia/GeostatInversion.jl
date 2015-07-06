using Optim
using PyPlot
using Calculus

close("all")

include("deconvolutionTestProblem.jl") 
A,strue,yvec,Gamma,C = deconv2(20,5);

#s0 = -0.5*strue + 0.1*randn(length(strue));
s0 = zeros(length(strue));

function cost(s::Vector)
  c = 0.5.*(yvec-A*s)'*inv(Gamma)*(yvec-A*s) + 0.5.*(s-s0)'*inv(C)*(s-s0)
  c = reshape(c,1)[1]
end

function cost_gradient!(x::Vector, y::Vector)
	result = gradient(cost, x)
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

res = optimize(cost, s0, iterations = 20,store_trace = true,)

x = linspace(0,1,20)
plot(x,strue,x,s0,x,res.minimum,linestyle="-",marker="o")

res = optimize(cost, s0, iterations = 50,store_trace = true,)
plot(x,res.minimum,linestyle="-",marker="o")


res = optimize(cost, s0, iterations = 100,store_trace = true,)
plot(x,res.minimum,linestyle="-",marker="o")

res = optimize(cost, s0, store_trace = true,)
plot(x,res.minimum,linestyle="-",marker="o")

legend(["synthetic","initial","s_20","s_50","s_100","min = s_$n"])





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

import Base.*
import Base.eltype
import Base.size
import Base.Ac_mul_B!
import Base.At_mul_B!
import Base.A_mul_B!
import Base.\
import Base.transpose

type LowRankCovMatrix
	samples::Array{Array{Float64, 1}, 1}
	function LowRankCovMatrix(samples::Array{Array{Float64, 1}, 1})
		zeromeansamples = Array{Array{Float64, 1}}(length(samples))
		means = zeros(Float64, length(samples[1]))
		for i = 1:length(samples)
			for j = 1:length(means)
				means[j] += samples[i][j]
			end
		end
		means /= length(samples)
		for i = 1:length(samples)
			zeromeansamples[i] = samples[i] - means
		end
		return new(zeromeansamples)
	end
end

type PCGALowRankMatrix
	etas::Array{Array{Float64, 1}, 1}
	HX::Array{Float64, 1}
	R
end

function transpose(A::LowRankCovMatrix)
	return A
end

function eltype(A::Union{PCGALowRankMatrix, LowRankCovMatrix})
	return Float64
end

function size(A::LowRankCovMatrix)
	return (length(A.samples[1]), length(A.samples[1]))
end

function size(A::LowRankCovMatrix, i::Int)
	if i == 1 || i == 2
		return length(A.samples[1])
	else
		error("there is no $i-th dimension in a LowRankCovMatrix")
	end
end

function size(A::PCGALowRankMatrix, i::Int)
	if i == 1 || i == 2
		return length(A.etas[1]) + 1#the +1 is for HX
	else
		error("there is no $i-th dimension in a PCGALowRankMatrix")
	end
end

function A_mul_B!(v::Vector, A::LowRankCovMatrix, x::Vector)
	fill!(v, 0.)
	for i = 1:length(A.samples)
		BLAS.axpy!(1. / (length(A.samples) - 1), A.samples[i] * dot(x, A.samples[i]), v)
	end
	return v
end

function A_mul_B!(v::Vector, A::PCGALowRankMatrix, x::Vector)
	xshort = x[1:end - 1]
	v[1:end - 1] = A.R * xshort
	v[end] = dot(A.HX, xshort)
	for i = 1:length(A.etas)
		dotp = dot(A.etas[i], xshort)
		for j = 1:length(v) - 1
			v[j] += A.etas[i][j] * dotp
		end
	end
	for j = 1:length(v) - 1
		v[j] += A.HX[j] * x[end]
	end
	return v
end

function Ac_mul_B!(v::Vector, A::Union{LowRankCovMatrix, PCGALowRankMatrix}, x::Vector)
	return A_mul_B!(v, A, x)#matrix is symmetric
end

function At_mul_B!(v::Vector, A::Union{LowRankCovMatrix, PCGALowRankMatrix}, x::Vector)
	return A_mul_B!(v, A, x)#A is symmetric
end

function *(A::PCGALowRankMatrix, x::Vector)
	result = Array{Float64}(size(A, 1))
	A_mul_B!(result, A, x)
	return result
end

function *(A::LowRankCovMatrix, B::Matrix)
	result = zeros(Float64, size(A, 1), size(B, 2))
	for i = 1:length(A.samples)
		BLAS.ger!(1. / (length(A.samples) - 1), A.samples[i], At_mul_B(B, A.samples[i]), result)
	end
	return result
end

function *(B::Matrix, A::LowRankCovMatrix)
	result = zeros(Float64, size(B, 1), size(A, 2))
	for i = 1:length(A.samples)
		BLAS.ger!(1. / (length(A.samples) - 1), B * A.samples[i], A.samples[i], result)
	end
	return result
end

function *(A::LowRankCovMatrix, x::Vector)
	result = zeros(Float64, size(A, 1))
	A_mul_B!(result, A, x)
	return result
end

function \(A::LowRankCovMatrix, b::Vector)
	result, c = IterativeSolvers.lsqr(A, b; maxiter=length(A.samples))
	return result
end

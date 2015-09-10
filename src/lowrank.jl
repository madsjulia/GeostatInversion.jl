import Base.*
import Base.eltype
import Base.size
import Base.Ac_mul_B!
import Base.A_mul_B!

type PCGALowRankMatrix
	etas::Array{Array{Float64, 1}, 1}
	HX::Array{Float64, 1}
	R
end

function eltype(A::PCGALowRankMatrix)
	return Float64
end

function size(A::PCGALowRankMatrix, i::Int)
	if i == 1 || i == 2
		return length(A.etas[1]) + 1#the +1 is for HX
	else
		error("there is no $i-th dimension in a PCGALowRankMatrix")
	end
end

function A_mul_B!(v::Vector, A::PCGALowRankMatrix, x::Vector)
	xshort = x[1:end - 1]
	v[1:end - 1] = A.R * xshort
	v[end] = dot(A.HX, xshort)
	for i = 1:length(A.etas)
		#v[1:end - 1] += A.etas[i] * dot(A.etas[i], x[1:end - 1])
		dotp = dot(A.etas[i], xshort)
		for j = 1:length(v) - 1
			v[j] += A.etas[i][j] * dotp
		end
	end
	return v
end

function Ac_mul_B!(v::Vector, A::PCGALowRankMatrix, x::Vector)
	return A_mul_B!(v, A, x)#matrix is symmetric
end

function *(A::PCGALowRankMatrix, x::Vector)
	result = Array(Float64, size(A, 1))
	A_mul_B!(result, A, x)
	return result
end

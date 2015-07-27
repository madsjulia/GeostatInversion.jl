import FiniteDifference2D
import FFTRF
import BlackBoxOptim
# using PyCall
# @pyimport matplotlib.pyplot as plt
using PyPlot

# Forward model, covariance matrix, and helper functions for the 2D
# groundwater example for PCGA, called by test.jl
# Dan O'Malley
# Last updated July 17, 2015 by Ellen Le
# Questions: omalled@lanl.gov, ellenble@gmail.com

srand(0)

const noise = 5 # what percent noise i.e. noise =5 means 5% of max value
# of yvec

const m = 20#the number of nodes on the pressure grid in the x-direction
const n = 20#the number of nodes on the pressure grid in the y-direction
#the domain we are solving over is (a,b)x(c,d)
const a = 0
const b = 1
const c = 0
const d = 1
#distance between points on the solution grid in the x and y directions
const hx = (b - a) / (m - 1)
const hy = (d - c) / (n - 1)
#the next 4 parameters are for the permeability fields
const mean_logk = 0.
const sigma_logk = 0.5
const betap = -3.5
f(x, y) = 0.#the "source"
#u_d(x, y) = 1 - y + x * (1 - y)
u_d(x, y) = 1 - y#the dirichlet boundary condition that is used on y=c and y=d
u_n(x, y) = 0.#the neumann boundary condition that is used on x=a, x=b
const numobs = 25#the number of observations
observationpoints = BlackBoxOptim.Utils.latin_hypercube_sampling([a, c], [b, d], numobs)#generate a bunch of observation points
observationI = Array(Int64, numobs)
observationJ = Array(Int64, numobs)
u_obs = Array(Float64, numobs)
k_obs = Array(Float64, numobs)
xy_obs = Array(Float64, (2, numobs))
#obsweights = ones(numobs) / (numobs)

#set everything up for a randomly generated k field with a power-law spectral density
function fftrflogk(; regweight1=1e-3, regweight2=1e-3)
	logk = FFTRF.powerlaw_structuredgrid([a - .5 * hx, c - .5 * hy], [b + .5 * hx, d + .5 * hy], [2 * m + 1, 2 * n + 1], mean_logk, sigma_logk, betap)'
	truelogk1 = Array(Float64, (m + 1, n))
	truelogk2 = Array(Float64, (m, n + 1))
	for i = 1:m + 1 
		for j = 1:n 
			truelogk1[i, j] = logk[2 * i - 1, 2 * j] #pts
                    #left and right of the pressure grid pts
		end
	end
	for i = 1:m
		for j = 1:n + 1 
			truelogk2[i, j] = logk[2 * i, 2 * j - 1] #pts
                    #above and below press grid pts
		end
	end
	regweight = 1 / (length(truelogk1) + length(truelogk2)) / std(logk)
	function forwardObsPoints(logx::Vector)
		u = forwardFullGrid(logx)
		result = Array(Float64, numobs)
		for i = 1:numobs
		#	result[i] = sqrt(obsweights[i]) *
                                                          # (u[observationI[i], observationJ[i]] - u_obs[i])
                    result[i] = u[observationI[i], observationJ[i]] 
		end
		return result
	end
	return logk, truelogk1, truelogk2, forwardObsPoints
end
logk, truelogk1, truelogk2, forwardObsPoints = fftrflogk()

function forwardFullGrid(logx::Vector)
	x = 10 .^ logx
	k1 = reshape(x[1:(m + 1)*n], (m + 1, n))
	k2 = reshape(x[(m + 1)*n+1:end], (m, n + 1))
	u = FiniteDifference2D.solveds(k1, k2, u_d, u_n, f, m, n, a, b, c, d)
	return u
end
#set up the true solution
trueu = forwardFullGrid([truelogk1[:]; truelogk2[:]])
#set up the observations, which are the "truth" at a number of
#observation points


for i = 1:numobs
	observationI[i] = 1 + int(round((m - 1) * (observationpoints[1, i] - a) / (b - a)))
	observationJ[i] = 1 + int(round((n - 1) * (observationpoints[2,i] - c) / (d - c)))
	u_obs[i] = trueu[observationI[i], observationJ[i]] 
	k_obs[i] = truelogk1[observationI[i], observationJ[i]] 
	xy_obs[1, i] = a + (observationI[i] - 1) * hx
	xy_obs[2, i] = c + (observationJ[i] - 1) * hy
end

#Add gaussian noise to perfect data vector at observation points u_obs
std_noise = maximum(abs(u_obs))*(noise/100)
k_std_noise = maximum(abs(k_obs))*(noise/100)
R = std_noise*eye(numobs)

srand(1234)
u_obsNoise = vec(u_obs + std_noise*randn(numobs,1))
k_obsNoise = vec(k_obs + k_std_noise*randn(numobs,1))


x0 = mean_logk * ones((m + 1) * n + m * (n + 1))

#here are a bunch of helper functions for plotting, etc

# # map log k to its actual coordinates to make a covariance matrix based
# # on Euclidean distance
function xyCoordsLogK(logx::Vector)

coords = Array((Float64,Float64), length(logx))
    l = 1
    #find the coords of the logk1 first
    for j = 1:n 
        for i = 1:m + 1 
	    coords[l] = ( -hx/2 + (i-1)*hx , (j-1)*hy )
            l = l+1 
	end
    end    
    #find the coord of the logk2
    for j = 1:n + 1     
        for i = 1:m
	    coords[l] = ( (i-1)*hx , -hy/2 + (j-1)*hy ) 
            l = l+1
	end
    end
# @bp    
return coords
end

logkvect = [truelogk1[:],truelogk2[:]]
coords = xyCoordsLogK(logkvect);

lenCoords = length(coords)

cov(h) = exp(-abs(h) / 0.3)

# Make the exponential isotropic covariance function Q
Q_up = Array(Float64, lenCoords, lenCoords)
#fill in the upper tri part of Q then copy it over since Q symmetric
for i = 1:lenCoords
    for j = (i+1):lenCoords
        dist = norm(collect(coords[i]) - collect(coords[j]))
        Q_up[i,j] = cov(dist)
    end
end

diagp = diagm(ones(lenCoords))

Q = Q_up + Q_up' + diagp


function x2k(x::Vector) #puts vectorized logkvect back into k1,k2 matrices
	k1 = reshape(x[1:(m + 1)*n], (m + 1, n))
	k2 = reshape(x[(m + 1)*n+1:end], (m, n + 1))
	return k1, k2
end

function ks2k(k1::Matrix, k2::Matrix) #puts the logk1, logk2 matrices
    #back into the full grid with invisible points, does an averaging at the missing points
	m = size(k2, 1)
	n = size(k1, 2)
	k = NaN * Array(Float64, (2 * m + 1, 2 * n + 1))
	for i = 1:m + 1
		for j = 1:n
			k[2 * i - 1, 2 * j] = k1[i, j]
		end
	end
	for i = 1:m
		for j = 1:n + 1
			k[2 * i, 2 * j - 1] = k2[i, j]
		end
	end
	for i = 1:size(k, 1)
		for j = 1:size(k, 2)
			if isnan(k[i, j])
				count = 0
				runningtotal = 0.
				if i != 1
					count += 1
					runningtotal += k[i - 1, j]
				end
				if i != size(k, 1)
					count += 1
					runningtotal += k[i + 1, j]
				end
				if j != 1
					count += 1
					runningtotal += k[i, j - 1]
				end
				if j != size(k, 2)
					count += 1
					runningtotal += k[i, j + 1]
				end
				k[i, j] = runningtotal / count
			end
		end
	end
	return k
end


function plotfield(field,totfignum,fignum,vmin,vmax)
    subplot(1,totfignum,fignum)
    imshow(transpose(field), extent=[c, d, a, b],
    interpolation="nearest")
    clim(vmin,vmax)
    for i = 1:numobs
	plot(observationpoints[1, i], observationpoints[2, i], ".", color="#E0B0FF")
    end
end



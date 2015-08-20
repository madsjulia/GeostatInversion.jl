    # blobV2.jl - optimized blob.jl
    # Forward model, covariance matrix, and helper functions for the 2D
    # groundwater example for PCGA, for black and white Log K fields with sharp boundaries
    # Dan O'Malley
    # Last updated 20 August 2015 by Ellen Le
    # Questions: omalled@lanl.gov, ellenble@gmail.com

import FiniteDifference2D
import FFTRF
import BlackBoxOptim
using PyPlot

@everywhere begin
    const m = 50#the number of nodes on the pressure grid in the x-direction
    const n = 50#the number of nodes on the pressure grid in the y-direction

    #the domain we are solving over is (a,b)x(c,d)
    const a = 0
    const b = 1
    const c = 0
    const d = 1

    #distance between points on the solution grid in the x and y directions
    const hx = (b - a) / (m - 1)
    const hy = (d - c) / (n - 1)
     
    const sqrtnumobs = 71
    const numobs = sqrtnumobs * sqrtnumobs
 
   #observationpoints = BlackBoxOptim.Utils.latin_hypercube_sampling([a, c], [b, d], numobs)#generate a bunch of observation points
    observationpoints = Array(Float64, (2, numobs))
    for i = 1:sqrtnumobs
	for j = 1:sqrtnumobs
	    observationpoints[1, i + (j - 1) * sqrtnumobs] = a + (b - a) * (i - .5) / sqrtnumobs
	    observationpoints[2, i + (j - 1) * sqrtnumobs] = c + (d - c) * (j - .5) / sqrtnumobs
	end
    end
    observationI = Array(Int64, numobs)
    observationJ = Array(Int64, numobs)
    u_obs = Array(Float64, numobs)
    #k_obs = Array(Float64, numobs)
    xy_obs = Array(Float64, (2, numobs))

    function bloblogk(; regweight2=1e-3, regweight3=1e2)
        const mean_logk = 1.
        const other_mean_logk = -1.
	logk = mean_logk * ones(2 * m + 1, 2 * n + 1)
	for i = 1:2 * m + 1
	    for j = 1:2 * n + 1
		if (i - m - 1) ^ 2 + (j - n - 1) ^ 2 < m * m / 4
		    logk[i, j] = other_mean_logk
		end
	    end
	end
	truelogk1 = Array(Float64, (m + 1, n))
	truelogk2 = Array(Float64, (m, n + 1))
	for i = 1:m + 1
	    for j = 1:n
		truelogk1[i, j] = logk[2 * i - 1, 2 * j]
	    end
	end
	for i = 1:m
	    for j = 1:n + 1
		truelogk2[i, j] = logk[2 * i, 2 * j - 1]
	    end
	end
	function forwardObsPoints(logx::Vector) #The forward map, calls ForwardFullGrid then picks out interp points
	    u = forwardFullGrid(logx)
	    result = Array(Float64, numobs)
	    for i = 1:numobs
                result[i] = u[observationI[i], observationJ[i]] 
	    end
	    return result
	end
	return logk, truelogk1, truelogk2, forwardObsPoints
    end

    logk, truelogk1, truelogk2, forwardObsPoints = bloblogk()

    f(x, y) = 0.#the "source"
    u_d(x, y) = 1 - y#the dirichlet boundary condition that is used on y=c and y=d
    u_n(x, y) = 0.#the neumann boundary condition that is used on x=a, x=b
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
	#k_obs[i] = truelogk1[observationI[i], observationJ[i]] 
	xy_obs[1, i] = a + (observationI[i] - 1) * hx
	xy_obs[2, i] = c + (observationJ[i] - 1) * hy
    end
end

#Add gaussian noise to perfect data vector at observation points u_obs
function make_yandR(u_obs::Vector,noise::Float64)
    std_noise = maximum(abs(u_obs))*(noise/100)
    # k_std_noise = maximum(abs(k_obs))*(noise/100)
    R = std_noise*eye(length(u_obs)) #assumption that datapoints have iid noise
    srand(1234)
    u_obsNoise = vec(u_obs + std_noise*randn(numobs,1))
    # k_obsNoise = vec(k_obs + k_std_noise*randn(numobs,1))
    return u_obsNoise,R
end

# x0 = mean_logk * ones((m + 1) * n + m * (n + 1)) #Not used right now

# Helper functions for plotting

# # map logkvect to its actual coordinates to make a covariance matrix based
# # on Euclidean distance
function xyCoordsLogK(logx::Vector)
    coords = fill(zeros(2),length(logx))
    l = 1
    #find the coords of the logk1 first
    for j = 1:n 
        for i = 1:m + 1 
	    coords[l] = [-hx/2 + (i-1)*hx, (j-1)*hy]
            l = l+1 
	end
    end    
    #find the coord of the logk2
    for j = 1:n + 1     
        for i = 1:m
	    coords[l] = [(i-1)*hx, -hy/2 + (j-1)*hy] 
            l = l+1
	end
    end
    return coords
end

function makeCovQ(logkvect::Vector,covdenom::Float64, alpha::Int64)
    lenCoords = length(logkvect)
    coords = xyCoordsLogK(logkvect)
    Q_lt = zeros(lenCoords, lenCoords)
    for j= 1:lenCoords
        for i = (j+1):lenCoords
            dist = norm(coords[i] - coords[j])
            Q_lt[i,j] = exp(-abs(dist) /covdenom)
        end
    end
    diagp = diagm(ones(lenCoords))
    Q = alpha*(Q_lt + Q_lt' + diagp)
    return Q
end


function x2k(x::Vector) #puts vectorized logkvect back into k1,k2 matrices
    k1 = reshape(x[1:(m + 1)*n], (m + 1, n))
    k2 = reshape(x[(m + 1)*n+1:end], (m, n + 1))
    return k1, k2
end

function ks2k(k1::Matrix, k2::Matrix) #puts the logk1, logk2 matricse
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


function plotfield(field,nrow,ncol,fignum,vmin,vmax;noObs=false, mycmap="jet")
    subplot(nrow,ncol,fignum)
    imshow(transpose(field), extent=[c, d, a, b],interpolation="nearest",cmap=mycmap)
    clim(vmin,vmax)
    if noObs == false
        for i = 1:numobs
	    plot(observationpoints[1, i], observationpoints[2, i], ".", color="#E0B0FF")
        end
    end
end



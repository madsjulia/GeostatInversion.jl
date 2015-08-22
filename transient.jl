import FiniteDifference2D
import FFTRF
import BlackBoxOptim
# using PyCall
# @pyimport matplotlib.pyplot as plt
using PyPlot

@everywhere begin
srand(0)

const covdenom = 0.2
const alpha = 20
const noise = 5 # what percent noise i.e. noise = 5 means 5% of max value of yvec
const m = 10#the number of nodes on the pressure grid in the x-direction
const n = 10#the number of nodes on the pressure grid in the y-direction
const finaltime = 1.#time at which the transient solution stops
const numtimesteps = 20
const deltat = finaltime / (numtimesteps - 1)
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
f_steady(x, y) = 0.#the "source"
#u_d(x, y) = 1 - y + x * (1 - y)
u_d(x, y) = 1 - y#the dirichlet boundary condition that is used on y=c and y=d
u_n(x, y) = 0.#the neumann boundary condition that is used on x=a, x=b
const sqrtnumobspoints = 5
const numobspoints = sqrtnumobspoints * sqrtnumobspoints
observationpoints = Array(Float64, (2, numobspoints))
for i = 1:sqrtnumobspoints
	for j = 1:sqrtnumobspoints
		observationpoints[1, i + (j - 1) * sqrtnumobspoints] = a + (b - a) * (i - .5) / sqrtnumobspoints
		observationpoints[2, i + (j - 1) * sqrtnumobspoints] = c + (d - c) * (j - .5) / sqrtnumobspoints
	end
end
observationI = Array(Int64, numobspoints)
observationJ = Array(Int64, numobspoints)
xy_obs = Array(Float64, (2, numobspoints))
u_obs = Array(Float64, numobspoints)
for i = 1:numobspoints
	observationI[i] = 1 + int(round((m - 1) * (observationpoints[1, i] - a) / (b - a)))
	observationJ[i] = 1 + int(round((n - 1) * (observationpoints[2,i] - c) / (d - c)))
	xy_obs[1, i] = a + (observationI[i] - 1) * hx
	xy_obs[2, i] = c + (observationJ[i] - 1) * hy
end

#set everything up for a randomly generated k field with a power-law spectral density
function fftrflogk()
	logk = FFTRF.powerlaw_structuredgrid([a - .5 * hx, c - .5 * hy], [b + .5 * hx, d + .5 * hy], [2 * m + 1, 2 * n + 1], mean_logk, sigma_logk, betap)'
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
	function forwardObsPoints(logx::Vector)
		x = 10 .^ logx
		k1, k2 = FiniteDifference2D.x2ks(x, m, n)
		result = Array(Float64, numobspoints * numobspoints * numtimesteps)
		u_0 = FiniteDifference2D.solveds(k1, k2, u_d, u_n, f_steady, m, n, a, b, c, d)[1:end]
		S = 1.
		for pumpingwellindex = 1:numobspoints
			#do a transient run with pumping at this obs point
			f_transient(x, y) = (x - xy_obs[1, pumpingwellindex]) ^ 2 + (y - xy_obs[2, pumpingwellindex]) ^ 2 < 1e-8 ? 1. : 0.
			u = FiniteDifference2D.solvetransient(k1, k2, S, u_0, u_d, u_n, f_transient, m, n, a, b, c, d, deltat, numtimesteps)
			for timestep = 1:numtimesteps
				for obspointindex = 1:numobspoints
					result[obspointindex + (timestep - 1) * numobspoints + (pumpingwellindex - 1) * numobspoints * numtimesteps] = u[observationI[obspointindex] + (observationJ[obspointindex] - 1) * m, timestep]
				end
			end
		end
		return result
	end
	return logk, truelogk1, truelogk2, forwardObsPoints
end
logk, truelogk1, truelogk2, forwardObsPoints = fftrflogk()

u_obs = forwardObsPoints([truelogk1[:]; truelogk2[:]])
end

#Add gaussian noise to perfect data vector at observation points u_obs
std_noise = maximum(abs(u_obs)) * (noise / 100)
R = std_noise * eye(length(u_obs)) #assumption that datapoints have iid noise
u_obsNoise = u_obs + std_noise * randn(length(u_obs))
x0 = mean_logk * ones((m + 1) * n + m * (n + 1))
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
	return coords
end

logkvect = [truelogk1[:],truelogk2[:]]
coords = xyCoordsLogK(logkvect);
lenCoords = length(coords)
cov(h) = alpha * exp(-abs(h) / covdenom)
Q = Array(Float64, lenCoords, lenCoords)# Make the exponential isotropic covariance function Q
#fill in the upper tri part of Q then copy it over since Q symmetric
for i = 1:lenCoords
	Q[i, i] = cov(0.)
	for j = (i+1):lenCoords
		dist = norm(collect(coords[i]) - collect(coords[j]))
		Q[i,j] = cov(dist)
		Q[j,i] = Q[i, j]
	end
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
        for i = 1:numobspoints
			plot(observationpoints[1, i], observationpoints[2, i], ".", color="#E0B0FF")
        end
    end
end

using PyPlot
using PyCall
@pyimport matplotlib.patches as mpatches

# Tan Bui-Thanh, April 2012
# Institute for computational engineering and sciences
# The University of Texas at Austin
# tanbui@ices.utexas.edu

# explore the posterior with smooth priors and without hyper-parameters.
#close("all")

function toeplitz(x)
    n = length(x);
    A = zeros(n, n);
    for i = 1:n
        A[i,i:end] = x[1:n - i + 1];
        A[i:end,i] = x[1:n - i + 1];
    end
    return A
end


srand(1)

# Mesh
n = 100; 
s = linspace(0,1,n+1); 
t = s;

# Prior flag
PriorFlag = 2; # 1: L_D Dirichlet 0
# 2: L_A


# discretize the deblurring kernel
beta = 0.05;
a = 1/sqrt(2*pi*beta^2)*exp(-0.5*(1/beta^2)*t.^2);
A = 1/n*toeplitz(a);

# Truth 
xtrue = 10*(t-0.5).*exp(-0.5*1e2*(t-0.5).^2) -0.8 + 1.6*t;

##------------additive noise-----------
noise = 5;       # Noise level in percentages of the max. of noiseless signal
y0 = A*xtrue;    # Noiseless signal
sigma = maximum(abs(y0))*(noise/100);                # STD of the additive noise
y = y0 + sigma*randn(n+1,1);


##------------Prior construction----------
# standard deviation of the innovation
gamma = 1/n;

# Construct the L_D matrix
if PriorFlag == 1
    L = diagm(ones(n+1)) - diagm(0.5*ones(n),1) - diagm(0.5*ones(n),-1);
elseif PriorFlag == 2
    L_D = diagm(ones(n+1)) - diagm(0.5*ones(n),1) - diagm(0.5*ones(n),-1);
    # you should never do this, but we do it anyway for convenience
    L_Dinv = inv(L_D);
    Dev = sqrt(gamma^2 * diag(L_Dinv * L_Dinv'));

    delta = gamma./ Dev[floor(n/2)];
    L = L_D; 
    L[1,:] = 0; 
    L[1,1] = delta;
    L[end,:] = 0; 
    L[end,end] = delta;
else
    println("example not supported")
end

# Calculating the MAP estimate and posterior variances, by least squares
xmean = [(1/sigma)*A;1/gamma*L]\[(1/sigma)*y;zeros(n+1,1)];
Gamma_post = inv((1/sigma^2)*A'*A + 1/gamma^2*L'*L); 

# Plotting the MAP estimate and the 2*STD envelope

STD = sqrt(diag(Gamma_post));
xhigh = xmean + 2*STD;
xlow = xmean - 2*STD;

figure()
p1 = plot(t,xmean,color = "red",linewidth=2)
p2 = plot(t,xtrue,color = "black",linewidth=1.5)
fill_between(t,vec(xlow),vec(xhigh),facecolor="powderblue")

b_patch = mpatches.Patch(color="powderblue")

legend([p1,p2,b_patch],["MAP", "truth","uncertainty"],loc="best")

relerrorPinv = norm(xmean-xtrue)/norm(xtrue)
@show(relerrorPinv)

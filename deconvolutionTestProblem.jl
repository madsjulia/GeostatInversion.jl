# Ellen Le June 12, 2015
# A deconvolution inverse problem to test our method on
# For more information see
# http://users.ices.utexas.edu/~tanbui/teaching/Bayesian/Bayesian.pdf

# using Debug

function toeplitz(x)
    n = length(x);
    A = zeros(n, n);
    for i = 1:n
        A[i,i:end] = x[1:n - i + 1];
        A[i:end,i] = x[1:n - i + 1];
    end
    return A
end


function deconv2(n,noise)
#.5 noise means STD of noise will be .005 of max ynot 

n=n-1; #if we want an n by n matrix representing n points on 0 to 1, has
    #n-1 intervals


# Mesh
t = linspace(0,1,n+1);
## discretize the deblurring kernel

beta = 0.05;
a = 1/sqrt(2*pi*beta^2)*exp(-0.5*(1/beta^2)*t.^2);
A = 1/n*toeplitz(a); #toeplitz is a diag matrix with a as diag, 1/n is
    #the width of each riemann rectangle


# Truth u = u(t), discretized at pts in the mesh, that we want to recover
utrue = -1.6*sin(2*pi*t);

## ------------Add noise to truth to create data-----------

y0 = A*utrue;    # Noiseless signal - true value in the absence of errorm
std = maximum(abs(y0))*(noise/100);  # STD of the noise for each data pt
Gamma = std*eye(n+1);  #covariance matrix is diagonal if we assume each
    #data pt is independent

srand(1234)
y = y0 + std*randn(n+1,1); #adds normally dist noise to each component
    #to get the data/y obs

#randn is draw w mean 0 and var 1
#so this is a sample with mean y_0 and variance std^2


## constructing the prior model
# we believe the truth function is smooth, with some uncertainty
gamma = 1/n;  #parameter - std of innovative term w - controls how
    #strong the prior is in the post model



# Construct the L matrix - C:= Cov(u) = (1/gamma^2)*inv(L'*L) since
# exponent in proba

# density is (1/gamma^2)*||L*(u-u0)||^2
L_D = diagm(ones(n+1)) - diagm(0.5*ones(n),1) - diagm(0.5*ones(n),-1);
  
# you should never do this, but we do it anyway for convenience
  L_Dinv = inv(L_D);
  Dev = sqrt(gamma^2 * diag(L_Dinv * L_Dinv'));

  delta = gamma./ Dev[floor(n/2)];
  
  L = L_D; 
  
 # for a prior where the endpoints are allowed to vary
  L[1,:] = 0; 
  L[1,1] = delta;
  
  L[end,:] = 0; 
  L[end,end] = delta;

  #the covariance matrix for our prior model
  #multiply times n to make C more important
  C = n *(gamma^2) * inv(L'*L);
 
    return A,utrue,y,Gamma,C
end

# A,u,y,Gamma,C = deconv2(10,5);
# Determistic inversion solution is 
# umap = (A'*inv(Gamma)*A+C)\(A*inv(Gamma)*y+C*u)

# julia> norm(umap-u)/norm(u)
# 0.007610802132033256


function testForward(u)
y = A*u;
return y
end
 
A,strue,yvec,Gamma,C = deconv2(20,5);





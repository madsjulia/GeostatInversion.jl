using PyPlot

#Plots the exponential decay versus 

x = linspace(0.05263,1.451,100)

denoms = [0.01,0.05,0.1,0.2,0.3,0.4]

figure()

#cov3(h) = (.002.*abs(h)).^3

for covdenom in denoms
    cov1(h) = exp(-abs(h) / covdenom)
    cov2(h) = exp(-h.^2 / covdenom.^2)
    
    plot(x,cov1(x),linestyle="-")
    plot(x,cov2(x),linestyle="--")
end

#plot(x,cov3(x),linestyle="--")  

xlabel("distance of transmissivity snodes")
ylabel("covariance")
title(L"Q with various $\sigma$, exponential in solid, Gaussian in dashed")
legend(["0.01","0.01","0.05","0.05","0.1","0.1","0.2","0.2","0.3","0.3","0.4","0.4"])

# #figure()
# for covdenom in denoms
   
# end
# 

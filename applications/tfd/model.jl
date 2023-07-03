#include("boson.jl")#
function theta(omega, beta)
    return atanh(exp(-beta * omega / 2.))
end

function V(alpha,omega,T)
    if T==0.0
        return alpha
    else
        return alpha*cosh(theta(omega,1.0/T))
    end
end

function Vtilde(alpha,omega,T)
    if T==0.0
        return 0.0
    else
        return alpha*sinh(theta(omega,1.0/T))
    end
end
    

function tfd_holstein(n;omega=1.0,t=1.0, alpha=1.0, T=0.0, order=["phys_bos","el","anc_bos"])
    os = OpSum()
    if T>0.0
        @assert order==["phys_bos","el","anc_bos"]
        ancpos=3
        stride=3
    else 
        @assert order=["phys_bos","el"]
        stride=2
    end
    elpos=2
    physpos=1
    for j in 1:(n - 1)
      elj=elpos
      
      os += -t,"Cdag", stride*(j-1)+elj, "C", stride*(j) +elj
      os += -t,"Cdag", stride*(j) +elj , "C", stride*(j-1)+elj
    end
    for j in 1:n
      if T>0.0
        os += Vtilde(alpha,omega,T),"n", stride*(j-1)+elpos, "A", stride*(j-1) + ancpos   #Vdag
        os += Vtilde(alpha,omega,T),"n", stride*(j-1)+elpos, "Adag", stride*(j-1) + ancpos
        os += -omega, "N", stride*(j-1)+ancpos #local ancilla oscillator
      end
      os += V(alpha,omega,T),"A", stride*(j-1) + physpos, "n", stride*(j-1)+elpos   #V
      os += V(alpha,omega,T), "Adag", stride*(j-1) + physpos, "n", stride*(j-1)+elpos
      os += omega, "N", stride*(j-1)+physpos #local oscillator
    end
    return os
  end

function zerotemp_holstein(n;omega=1.0,t=1.0, alpha=1.0, order=["phys_bos","el"])
    return tfd_holstein(n;omega=omega,t=t, alpha=alpha, T=0.0, order=order)
  end

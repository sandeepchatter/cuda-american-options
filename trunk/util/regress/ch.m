% This example's performance or consistency with the
% language good practices are not guaranteed.
function ch = chebyshev(n,x)
  switch n
      case 0
          ch=1;
      case 1
          ch=x;
      otherwise
          ch=2.*x.*chebyshev(n-1,x)-chebyshev(n-2,x);
 
  end
end

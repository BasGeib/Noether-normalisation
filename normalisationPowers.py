
import sympy as sp

x,y,z,u,v,w,e,k = sp.symbols('x y z u v w e k')

f = x*(x-z)*y+1
fs = f.subs([(z, v), (x, u+v**e), (y, w+v**k)])
fs.expand()
fs.subs([(e,0), (k,1)]).expand()
# Check which e, k work


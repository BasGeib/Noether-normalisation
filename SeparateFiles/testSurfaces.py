import sympy as sp

x, y, z, u, v, w = sp.symbols('x y z u v w')

f = x**2*z**2 - x*z**3 + y**2*z**2 - y*z**3 - 1

variables = {'x': x, 'y': y, 'z': z}

for name, as_v in variables.items():
    print(f"\n testing: v = {name}")
    
    if as_v == z:
        sub_rules = lambda e_val, k_val: [(z, v), (x, u + v**e_val), (y, w + v**k_val)]
    elif as_v == x:
        sub_rules = lambda e_val, k_val: [(x, v), (z, u + v**e_val), (y, w + v**k_val)]
    else:
        sub_rules = lambda e_val, k_val: [(y, v), (x, u + v**e_val), (z, w + v**k_val)]
        
    found_valid_linear = False
    
    for e_val in [0, 1]:
        for k_val in [0, 1]:
            fs = f.subs(sub_rules(e_val, k_val)).expand()
            poly_v = sp.Poly(fs, v)
            lc = poly_v.LC()
            
            if lc.is_number:
                print(f"succeeded with low exponent: e={e_val}, k={k_val}, Leading coeff: {lc}")
                found_valid_linear = True
                
    if not found_valid_linear:
        print(f"succes: Linear substitutions failed for v = {name}. ")
        
        discovered = False
        for e_val in range(1, 5):
            for k_val in range(1, 5):
                fs = f.subs(sub_rules(e_val, k_val)).expand()
                poly_v = sp.Poly(fs, v)
                if poly_v.LC().is_number:
                    print(f" First working pair: e={e_val}, k={k_val} (Max degree of v: {poly_v.degree()})")
                    discovered = True
                    break
            if discovered: break

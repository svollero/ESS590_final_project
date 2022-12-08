import numpy as np
from numpy.linalg import eig, inv, svd
from math import atan2

#Data manipulation
def matrix2xyz(matrix):
    m = np.tile(range(0,np.shape(matrix)[1]),[np.shape(matrix)[0],1])
    n = np.transpose(np.tile(range(0,np.shape(matrix)[0]),[np.shape(matrix)[1],1]))
    xyz = np.transpose([n[matrix==matrix],m[matrix==matrix],matrix[matrix==matrix]])
    return (xyz);
    
def add_field(dataframe, fieldvalues, fieldname):
    dataframe[fieldname] = fieldvalues
    return (dataframe)

#Ellipse fitting functions
def fit_ellipse(x,y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return (a)

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return (np.array([x0, y0]))

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    phi = atan2(2 * b, (a - c)) / 2
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return (phi)

def ellipse_axis_length( a ):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return (np.array([np.max([res1, res2]), np.min([res1, res2])]))

#Geometrical parameters functions
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi>=np.pi*2:
        phi=phi-np.pi*2
    if phi<0:
        phi=phi+np.pi*2
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def orintation_uv(orientation):
    u, v = pol2cart(np.ones(np.shape(orientation)[0]), np.deg2rad(orientation))
    return (u, v)

def elongation(long_axis, short_axis):
    el = short_axis/long_axis
    return (el)

def equivalent_diameter(clast_surface_area):
    eqd = 2*np.sqrt(clast_surface_area/np.pi)
    return (eqd)

def circularity(equivalent_diameter, long_axis):
    cir = equivalent_diameter/long_axis
    return (cir)

def sorting(sizes):
    sor = -np.log2(sizes/0.001)
    return (sor)

#Hydrodynamical parameters functions
def van_rijn_dimensionless_diameter(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    mu = water_dynamic_viscosity
    nu = mu / rho_water
    Rd = rho_sed / rho_water
    d_star = sizes*((g*(Rd-1))/(nu**2))**(1/3)
    return (d_star)

def soulsby_critical_shields(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    d_star = van_rijn_dimensionless_diameter(sizes, rho_water, rho_sed, g, water_dynamic_viscosity)
    theta = (0.24/d_star)+(0.055*(1-np.exp(-0.02*d_star)))
    theta[d_star<=5] = (0.30/(1+(1.2*d_star[d_star<=5])))+(0.055*(1-np.exp(-0.02*d_star[d_star<=5])))
    return (theta)
	
def soulsby_whitehouse_critical_shields(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    d_star = van_rijn_dimensionless_diameter(sizes, rho_water, rho_sed, g, water_dynamic_viscosity)
    theta = (0.3/(1+(1.2*d_star))) + (0.055*(1-np.exp(-0.02*d_star)))
    return (theta)

def shields_critical_shear_stress(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    theta = soulsby_critical_shields(sizes, rho_water, rho_sed, g, water_dynamic_viscosity)
    tau = theta* (rho_sed - rho_water) * g * sizes
    return (tau)

def shields_critical_shear_velocity(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    tau = shields_critical_shear_stress(sizes, rho_water, rho_sed, g, water_dynamic_viscosity)
    u_star = (tau / rho_water) ** (1/2)
    return (u_star)

def shields_critical_grain_reynolds_number(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    u_star = shields_critical_shear_velocity(sizes, rho_water, rho_sed, g, water_dynamic_viscosity)
    mu = water_dynamic_viscosity
    nu = mu / rho_water
    Re_star = (u_star*sizes) / nu
    return (Re_star)

def hjulstrom_deposition_velocity(sizes):
    v_dep = 77*(sizes/(1+(24*sizes)))
    return (v_dep)

def hjulstrom_motion_initiation_velocity(sizes, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    mu = water_dynamic_viscosity
    nu = mu / rho_water
    Rd = rho_sed / rho_water
    v_ero = 1.5*((nu/sizes)**0.8) + 0.85*((nu/sizes)**0.35) + 9.5*((Rd*g*sizes)/(1+(2.25*Rd*g*sizes)))
    return (v_ero)

def leroux_wave_orbital_velocity(sizes, wave_period, rho_water=1.025, rho_sed=2.600, g=9.81, water_dynamic_viscosity=1.07e-3):
    mu = water_dynamic_viscosity*1e1                                               #g/cm3
    g = g*1e2                                                                      #cm/s2
    D = sizes*1e2                                                                  #cm
    rho_gam = (rho_sed - rho_water)
    Dd = D*(((rho_water*g*rho_gam)/(mu**2)))**(1/3)                                #Leroux's dimensionless grain size
    Wd = np.zeros(np.shape(Dd))
    Wd[Dd<=1.2538] = (0.2354*Dd[Dd<=1.2538])**2
    Wd[Dd>1.2538] = (0.208*Dd[Dd>1.2538]-0.0652)**(3/2)
    Wd[Dd>2.9074] = (0.2636*Dd[Dd>2.9074]-0.37)
    Wd[Dd>22.9866] = (0.8255*Dd[Dd>22.9866]-5.4)**(2/3)
    Wd[Dd>134.92150] = (2.531*Dd[Dd>134.92150]+160)**(1/2)
    Wd[Wd==0] = np.nan
    theta_wl = (0.0246*Wd)**(-0.55)
    Uw = ((theta_wl*g*D*rho_gam)/(((np.pi*wave_period)/(rho_water*mu))**0.5))/1e2 #m/s
    return (Uw)
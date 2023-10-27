def twd97_to_lonlat(x=174458.0,y=2525824.0):
    import math
    """
    Parameters
    ----------
    x : float
        TWD97 coord system. The default is 174458.0.
    y : float
        TWD97 coord system. The default is 2525824.0.
    Returns
    -------
    list
        [longitude, latitude]
    """
    
    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0
    
    e = math.pow((1-math.pow(b, 2)/math.pow(a,2)), 0.5)
    
    x -= dx
    y -= dy
    
    M = y / k0
    
    mu = M / ( a*(1-math.pow(e, 2)/4 - 3*math.pow(e,4)/64 - 5 * math.pow(e, 6)/256))
    e1 = (1.0 - pow((1   - pow(e, 2)), 0.5)) / (1.0 +math.pow((1.0 -math.pow(e,2)), 0.5))
    
    j1 = 3*e1/2-27*math.pow(e1,3)/32
    j2 = 21 * math.pow(e1,2)/16 - 55 * math.pow(e1, 4)/32
    j3 = 151 * math.pow(e1, 3)/96
    j4 = 1097 * math.pow(e1, 4)/512
    
    fp = mu + j1 * math.sin(2*mu) + j2 * math.sin(4* mu) + j3 * math.sin(6*mu) + j4 * math.sin(8* mu)
    
    e2 = math.pow((e*a/b),2)
    c1 = math.pow(e2*math.cos(fp),2)
    t1 = math.pow(math.tan(fp),2)
    r1 = a * (1-math.pow(e,2)) / math.pow( (1-math.pow(e,2)* math.pow(math.sin(fp),2)), (3/2))
    n1 = a / math.pow((1-math.pow(e,2)*math.pow(math.sin(fp),2)),0.5)
    d = x / (n1*k0)
    
    q1 = n1* math.tan(fp) / r1
    q2 = math.pow(d,2)/2
    q3 = ( 5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1,2) - 9 * e2 ) * math.pow(d,4)/24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * math.pow(t1,2) - 3 * math.pow(c1,2) - 252 * e2) * math.pow(d,6)/720
    lat = fp - q1 * (q2 - q3 + q4)
    
    
    q5 = d
    q6 = (1+2*t1+c1) * math.pow(d,3) / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1,2) + 8 * e2 + 24 * math.pow(t1,2)) * math.pow(d,5) / 120
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)
    
    lat = (lat*180) / math.pi
    lon = (lon*180) / math.pi
    return [lon, lat]

def lonlat_to_97(lon,lat):
    """
    It transforms longitude, latitude to TWD97 system.

    Parameters
    ----------
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees 

    Returns
    -------
    x, y [TWD97]
    """
    import math
    lat = math.radians(lat)
    lon = math.radians(lon)
    
    a = 6378137.0
    b = 6356752.314245
    long0 = math.radians(121)
    k0 = 0.9999
    dx = 250000

    e = (1-b**2/a**2)**0.5
    e2 = e**2/(1-e**2)
    n = (a-b)/(a+b)
    nu = a/(1-(e**2)*(math.sin(lat)**2))**0.5
    p = lon-long0

    A = a*(1 - n + (5/4.0)*(n**2 - n**3) + (81/64.0)*(n**4  - n**5))
    B = (3*a*n/2.0)*(1 - n + (7/8.0)*(n**2 - n**3) + (55/64.0)*(n**4 - n**5))
    C = (15*a*(n**2)/16.0)*(1 - n + (3/4.0)*(n**2 - n**3))
    D = (35*a*(n**3)/48.0)*(1 - n + (11/16.0)*(n**2 - n**3))
    E = (315*a*(n**4)/51.0)*(1 - n)

    S = A*lat - B*math.sin(2*lat) + C*math.sin(4*lat) - D*math.sin(6*lat) + E*math.sin(8*lat)

    K1 = S*k0
    K2 = k0*nu*math.sin(2*lat)/4.0
    K3 = (k0*nu*math.sin(lat)*(math.cos(lat)**3)/24.0) * (5 - math.tan(lat)**2 + 9*e2*(math.cos(lat)**2) + 4*(e2**2)*(math.cos(lat)**4))

    y_97 = K1 + K2*(p**2) + K3*(p**4)

    K4 = k0*nu*math.cos(lat)
    K5 = (k0*nu*(math.cos(lat)**3)/6.0) * (1 - math.tan(lat)**2 + e2*(math.cos(lat)**2))

    x_97 = K4*p + K5*(p**3) + dx
    return x_97, y_97
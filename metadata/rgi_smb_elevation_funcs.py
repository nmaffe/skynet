import numpy as np
import pandas as pd
"Functions that map elevation to smb for each rgi"

def smb_elev_functs(rgi, elev, lat, lon):

    if rgi in [1,2]:
        a = -2.06716100e-03
        c = 1.00007291e+02
        p = 5.35472753e-01
        return a / (c + elev) ** p

    elif rgi in [3,4]:
        m = 3.55847897e-08
        q = -4.09376228e-05
        return m * elev + q

    elif rgi in [5]:
        m = np.nan
        q = np.nan
        return m * elev + q

    elif rgi in [6,]:
        m, q = 9.79209658e-08, -1.18727280e-04
        return m * elev + q

    elif rgi in [7,9]:
        m, q = 6.24861787e-08, -3.13537544e-05
        return m * elev + q

    elif rgi in [8,]:
        m, q = 1.28063292e-08, -2.91968790e-05
        return m * elev + q

    elif rgi in [10]:

        # mask 1
        if (((lon>150.) & (lat<67.)) | ((lon<-170.) & (lat<68.))):
            m, q = 2.73664257e-08, -5.57532056e-05

        # mask 2
        elif (lat > 70.5):
            m, q = 3.20075411e-09, -1.69586219e-05

        # mask 3
        elif (((lon > 123.) & (lon < 150.)) & ((lat > 59.) & (lat < 71.))):
            m, q = 8.98905066e-09, -4.55599936e-05

        # mask 4
        elif (((lon > 55.) & (lon < 98.)) & ((lat > 62.) & (lat < 71.))):
            m, q = 2.93816867e-08, -4.90286962e-05
        # mask 5
        elif (((lon > 83.) & (lon < 121.)) & ((lat > 44.) & (lat < 60.))):
            m, q = 1.20851920e-08, -7.95458401e-05

        else:
            raise ValueError(f"Region 10 point at which you want smb outside the 5 masks defined in smb study.")

        return m * elev + q

    elif rgi in [11]:
        m, q = 3.35388079e-08, -1.09618747e-04
        return m * elev + q

    elif rgi in [12]:
        m, q = 3.37561294e-08, -1.34817237e-04
        return m * elev + q

    elif rgi in [13]:
        m, q = 7.52278648e-10, -7.91066609e-06
        return m * elev + q

    elif rgi in [14]:
        m, q = 2.66700218e-09, -2.03540077e-05
        return m * elev + q

    elif rgi in [15]:
        m, q = 3.13560412e-09, -4.80703842e-05
        return m * elev + q

    elif rgi in [16]:
        # mask 1: Africa and Indonesia
        if (lon > -20.0):
            m, q = 1.25561251e-06, -2.45823803e-03

        # mask 2: north
        elif ((lon < -60.0) & (lat > -4.0)):
            m, q = 8.61184690e-08, -2.96609257e-04

        # mask 3: south west
        elif ((lon < -74.0) & (lat < -4.0)):
            m, q = 4.86895515e-08, -2.73268834e-04

        # mask 4: south east
        elif ((lon > -74.0) & (lon < -60.0) & (lat < -4.0)):
            m, q = 0.0, -1.48720385e-04

        return m * elev + q

    elif rgi in [17]:
        # mask 1: south
        if (lat <= -52.0):
            m, q = 1.94213019e-07, -1.50282861e-04
        # mask 2: north
        elif (lat > -52.0):
            m, q = 1.08078913e-08, -8.79352844e-05
        return m * elev + q

    elif rgi in [18]:
        m, q = 0.0, -2.47735564e-05
        return m * elev + q

    elif rgi in [19]:
        m = np.nan
        q = np.nan
        return m * elev + q

    else:
        raise ValueError(f"rgi value {rgi} does not exist.")

def smb_elev_functs_hugo(rgi=None):
    """
    returns rate of change of hugonnet dmdtda wrt elevation
    - m [d(dmdtda) / dh] = [m w.e./year / m a.s.l]
    - q [dmdtda] = [m w.e./year]
    """
    df = pd.DataFrame(index=range(1, 20), columns=['m', 'q'])
    df.loc[1, ['m', 'q']] = [0.0004589, -1.03817] # updated
    df.loc[2, ['m', 'q']] = [0.000335, -1.0190] # updated
    df.loc[3, ['m', 'q']] = [0.000216, -0.4851] # updated
    df.loc[4, ['m', 'q']] = [0.00065118, -0.87902] # updated
    df.loc[5, ['m', 'q']] = [0.0005457, -0.70308] # updated
    df.loc[6, ['m', 'q']] = [0.000837, -1.08213] # updated
    df.loc[7, ['m', 'q']] = [0.000863, -0.524328] # updated
    df.loc[8, ['m', 'q']] = [0.000562, -1.08825] # updated
    df.loc[9, ['m', 'q']] = [0.0006414, -0.40542] # updated
    df.loc[10, ['m', 'q']] = [0.0003031, -1.03351] # updated
    df.loc[11, ['m', 'q']] = [0.00041039, -1.73905] # updated
    df.loc[12, ['m', 'q']] = [0.0001430, -0.91881] # updated
    df.loc[13, ['m', 'q']] = [0.0004552, -1.9562] # updated
    df.loc[14, ['m', 'q']] = [0.0001617, -0.91912] # updated
    df.loc[15, ['m', 'q']] = [0.0003172, -2.05440] # updated
    df.loc[16, ['m', 'q']] = [0.0005212, -2.88924] # updated
    df.loc[17, ['m', 'q']] = [0.0004511, -1.0511] # updated
    df.loc[18, ['m', 'q']] = [0.0001404, -0.44008] # updated
    df.loc[19, ['m', 'q']] = [0.0004052, -0.19093] # updated

    return df


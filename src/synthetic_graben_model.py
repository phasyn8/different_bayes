import gempy as gp
import numpy as np

class well_from_graben_model(object):
    
    ''' This class has only one job and this is to toss out synthetic well logs for a gempy graben model (Lisa's)
    Density (RHO) and Porosity (PHI) are populated based on the lithology codes from the vertical borings
    
    Pro Tip!
    The compute_model function will bork without any additional values populated in the surface values, for reasons 
    that I am not sure of... some additonal features are built in the model, like sections and this was due to 
    previous attempts at creating well logs from 2D sections, which are still being worked with.
    
    Parameters:
        x : map section x position of sample (boring)
        y : map section y position of sample (boring)
        z_top : This is the top of the boring (nearer to the surface)
        z_bottom : This is the bottom of the boring (near the 'basement')
        res : Resolution of the log (number of equidistant samples)
        Rnoise : RHO (density) noise sigma (stdDev) for the convolution step (random normal, using the true value as mu)
        Pnoise : PHI (porosity) noise sigma
    '''
    
    def __init__(self, x, y, z_top, z_bottom, res, Rnoise, Pnoise):
        
        self.x = x
        self.y = y
        self.z_top = z_top
        self.z_bottom = z_bottom
        self.res = res
        self.Rnoise = Rnoise
        self.Pnoise = Pnoise
        #self.return_well = self.return_well(self.wellpath, self.welldata)
        #self.phi_rho = self.phi_rho
        #self.build_1d_well_path = self.build_1d_well_path
        
        data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

        geo_model = gp.create_data('viz_3d',
                           [0, 2000, 0, 2000, 0, 1600],
                           [50, 50, 50],
                           path_o=data_path + "data/input_data/lisa_models/foliations" + str(
                               7) + ".csv",
                           path_i=data_path + "data/input_data/lisa_models/interfaces" + str(
                               7) + ".csv"
                                )

        gp.map_stack_to_surfaces(geo_model,
                                {"Fault_1": 'Fault_1', "Fault_2": 'Fault_2',
                                 "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')}
                                )
        geo_model.add_surface_values([[0,0,.2, .3, .4, .2, .1, 0.03, .0], [0,0,2.1, 2, 1.8, 2.1, 2.4, 2.5, 2.9]], ['porosity', 'bulk_density'])
        geo_model.set_is_fault(['Fault_1', 'Fault_2'])
        section_dict = {'well1': ([100, 500], [100, 1100], [20, 20]), 'well2': ([10,2000],[2000,10],[100,100])
                }
        geo_model.set_section_grid(section_dict)
        #gp.plot.plot_section_traces(geo_model)
        geo_model.get_active_grids()
        #gp.set_interpolator(geo_model, theano_optimizer='fast_compile')
        #geo_model.get_active_grids()
        self.well_path = self.build_1d_well_path(x, y, z_top, z_bottom, res)
        print(self.well_path[:4])
        geo_model.get_active_grids()
        
        gp.set_interpolator(geo_model, theano_optimizer='fast_compile')
        
        sol = gp.compute_model(geo_model, at=self.well_path)
        self.well_data = self.phi_rho(sol, self.Rnoise, self.Pnoise)
        print(self.well_data[:1])
        #self.return_well(self.well_path, self.well_data)
        
        #return self.well_path, self.well_data
    #if __name__ == '__main__':
    
    def build_1d_well_path(self, x, y, z_top, z_bottom, res): #z_bottom and z_top are relative to a lower model boundary (basement) zero point, which puts the surface closer to zero 
        x = np.ones(res)*x
        y = np.ones(res)*y
        z = np.linspace(z_bottom , z_top, res)
        #well_path1 = np.empty(3)

        for i in range(res):
            if i == 0:
                well_path= np.array((x[0],y[0],z[0]))
            else:
                well_path = np.vstack((well_path, np.array((x[i],y[i],z[i]))), dtype=object)
        return well_path

    def phi_rho(self, sol, Rnoise, Pnoise):
        phi = []
        rho = []
        phiRho = np.vstack(([0,0,0,.2, .3, .4, .2, .1, 0.03, .0], [0,0,0,2.1, 2, 1.8, 2.1, 2.4, 2.5, 2.9]))
        #phiRho[1,4]
        for i in sol.custom[0][0]: 
            i = int(i)
            phi.append(phiRho[0,i])
            rho.append(phiRho[1,i])
        phiNoise = np.random.normal(phi, Pnoise)
        rhoNoise = np.random.normal(rho, Rnoise)
        return phi, phiNoise, rho, rhoNoise
    
    def return_well(self):
        return self.well_path, self.well_data
    
        
        
        
    
import numpy as np


'''
1D Reaction diffusion
'''

def laplacian1D(a, dx):
    return (
        - 2 * a
        + np.roll(a,1,axis=0) 
        + np.roll(a,-1,axis=0)
    ) / (dx ** 2)

def random_initialiser(shape):
    return(
        np.random.normal(loc=0, scale=0.05, size=shape),
        np.random.normal(loc=0, scale=0.05, size=shape)
    )

def random_periodic(shape):
    return(
        0.2*np.sin(np.linspace(0,12*3.14,shape))+ 0.8*np.sin(np.linspace(0,5*3.14,shape))+ 0.4*np.sin(np.linspace(0,10*3.14,shape)),
        0.8*np.sin(np.linspace(0,5*3.14,shape))- 0.1*np.sin(np.linspace(0,10*3.14,shape))
    )

class OneDimensionalRDEquations():
    def __init__(self, Da, Db, Ra, Rb,
                 initialiser=random_initialiser,
                 width=1000, dx=1, 
                 dt=0.1, steps=1):
        
        self.Da = Da
        self.Db = Db
        self.Ra = Ra
        self.Rb = Rb
        
        self.Ya = []
        self.Yb = []
        self.X = []
        self.initialiser = initialiser
        self.width = width
        self.dx = dx
        self.dt = dt
        self.steps = steps
        
    def initialise(self):
        self.t = 0
        self.a, self.b = self.initialiser(self.width)
    
    def make_grid(self):
        self.x_arr = np.arange(0,self.width,self.dx)
        self.t_arr = self.dt*np.arange(0,self.steps)
        self.x_grid, self.t_grid = np.meshgrid(self.x_arr, self.t_arr, indexing='ij')
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        
        # unpack so we don't have to keep writing "self"
        a,b,Da,Db,Ra,Rb,dt,dx = (
            self.a, self.b,
            self.Da, self.Db,
            self.Ra, self.Rb,
            self.dt, self.dx
        )
        
        La = laplacian1D(a, dx)
        Lb = laplacian1D(b, dx)
        
        delta_a = dt * (Da * La + Ra(a,b))
        delta_b = dt * (Db * Lb + Rb(a,b))

        self.a += delta_a
        self.b += delta_b     
        
        self.Ya.append(np.array(self.a))
        self.Yb.append(np.array(self.b))
        
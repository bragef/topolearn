
from .simpcomplex import SimplicalComplex, HomologyPairs
from .rips import RipsComplex, calc_distance_matrix


class TopologicalLoss:

    def __init__(self, filtration):
        self.filtration = filtration
        self.count_1 = 0
        self.count_2 = 0

    def find_simplices(A, max_dim = 1):

        simplices = self.filtration.fit(A)
        bdpairs =  simplices.birth_death_pairs()

        # Index get the simplices from 

        edges = []
        for b, d, dim, b_value, f_value  in bdpairs:
            # Diagonal values are just noise
            if b_value == f_value: 
                continue 

            # 0-dim, only death values are relevant    
            if dim == 0:
                edges.append(frozenset({(b, d)}))
            else:
                # Birth-edges
                s_dim, s_birth,  = simplices.get_simplex(b)


                

                









    def calculate_topo_loss(self, X, Z):
    
        Ax = calc_distance_matrix(X)
        Az = calc_distance_matrix(Z)

        simplices_x = self.filtration.fit(Ax)
        simplices_z = self.fitlration.fit(Az)
            
        bdpairs_x = simplices_x.birth_death_pairs()
        bdpairs_z = simplices_z.birth_death_pairs()

        # Find the _relevant edges_ in each set: 
        





        bdpairs_z = HomologyPairs(self.filtration.fit_and_transform(s_pred))
        
        # The birth-death pair are all assigned a birth and death distance, which correspond to
        # the length of the simplex which which create and destroy the simplex. 
        a1 = bdpairs_x.as_dict(0)
        a2 = bdpairs_x.as_dict(1)
        a3 = bdpairs_z.as_dict(0)
        a4 = bdpairs_z.as_dict(1)


        return bdpairs_x, bdpairs_z, a1, a2, a3, a4












    def __call__(self, input, output):
        self.count_1 += 1
        def loss():
            self.count_2 += 1
            print("Ahoy")
            return 0.5

        return loss











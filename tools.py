import numpy as np
from tqdm import tqdm

class IEN_class():
    def __init__(self, mesh, data_type, data, data_tag) -> None:
        # Possible data_types: Tet, Tri, Line
        self.mesh = mesh
        self.data_type = data_type
        self.data = data

        if type(data_tag) == str:
            self.data_tag = np.full(data.shape[0], data_tag)
        else:
            self.data_tag = data_tag

    def setDataTag(self, data_tag):
        self.data_tag = data_tag

    def __str__(self):
        return ("IEN of class %s with %i elements" % (self.data_type, self.data.shape[0]))

class IEN_domain_class(IEN_class):
    def IENFunc(self, K, M):
        if self.data_type == 'Line':
            self.IENLineFunc(K, M, self.data)
        elif self.data_type == 'Tri':
            self.IENTriFunc(K, M, self.data)
        elif self.data_type == 'Tet':
            self.IENTetFunc(K, M, self.data)

    #form functions for 2D
    def IENTetFunc(self, K, M, IEN) -> None:
        X = self.mesh.X
        Y = self.mesh.Y
        Z = self.mesh.Z

        for IENelem in tqdm(IEN):
            tet_matrix = [[1, X[IENelem[i]], Y[IENelem[i]], Z[IENelem[i]]] for i in range(4)]

            tet_volume = (1/6)*np.linalg.det(tet_matrix)

            [a_list, b_list, c_list, d_list] = np.linalg.inv(tet_matrix)

            melem = (tet_volume/20)*np.array(([2, 1, 1, 1],
                                              [1, 2, 1, 1],
                                              [1, 1, 2, 1],
                                              [1, 1, 1, 2]))

            kxelem = tet_volume*np.array([[b_list[i]*b_list[j] for i in range(4)] for j in range(4)])
            kyelem = tet_volume*np.array([[c_list[i]*c_list[j] for i in range(4)] for j in range(4)])
            kzelem = tet_volume*np.array([[d_list[i]*d_list[j] for i in range(4)] for j in range(4)])

            kelem = kxelem + kyelem + kzelem

            for ilocal in range(4):
                iglobal = IENelem[ilocal]

                for jlocal in range(4):
                    jglobal = IENelem[jlocal]

                    K[iglobal,jglobal] += kelem[ilocal,jlocal]
                    M[iglobal,jglobal] += melem[ilocal,jlocal]

    #form functions for 2D
    def IENTriFunc(self, K, M, IEN) -> None:
        X = self.mesh.X
        Y = self.mesh.Y

        for IENelem in tqdm(IEN):
            tri_matrix = [[1, X[IENelem[i]], Y[IENelem[i]]] for i in range(3)]

            tri_area = (1/2)*np.linalg.det(tri_matrix)

            [a_list, b_list, c_list] = np.linalg.inv(tri_matrix)

            melem = (tri_area/12)*np.array(([2, 1, 1],
                                            [1, 2, 1],
                                            [1, 1, 2]))

            kxelem = tri_area*np.array([[b_list[i]*b_list[j] for i in range(3)] for j in range(3)])
            kyelem = tri_area*np.array([[c_list[i]*c_list[j] for i in range(3)] for j in range(3)])

            kelem = kxelem + kyelem

            for ilocal in range(3):
                iglobal = IENelem[ilocal]

                for jlocal in range(3):
                    jglobal = IENelem[jlocal]

                    K[iglobal,jglobal] += kelem[ilocal,jlocal]
                    M[iglobal,jglobal] += melem[ilocal,jlocal]

    #form functions for 2D
    def IENLineFunc(self, K, M, IEN) -> None:
        X = self.mesh.X

        for IENelem in tqdm(IEN):
            line_matrix = [[1, X[IENelem[i]]] for i in range(2)]

            line_length = np.linalg.det(line_matrix)

            [a_list, b_list] = np.linalg.inv(line_matrix)

            melem = (line_length/6)*np.array([[2, 1],
                                              [1, 2]])

            kxelem = line_length*np.array([[b_list[i]*b_list[j] for i in range(2)] for j in range(2)])

            kelem = kxelem

            for ilocal in range(2):
                iglobal = IENelem[ilocal]

                for jlocal in range(2):
                    jglobal = IENelem[jlocal]

                    K[iglobal,jglobal] += kelem[ilocal,jlocal]
                    M[iglobal,jglobal] += melem[ilocal,jlocal]

class IEN_bc_class(IEN_class):
    def __init__(self, mesh, data_type, data, data_tag, prescription) -> None:
        super().__init__(mesh, data_type, data, data_tag)
        self.prescription = prescription

    def setPrescription(self, prescription):
        if prescription.shape[0] == self.data.shape[0]:
            self.prescription = prescription
        else:
            print("Prescription does not have correct shape")

    def retrieveBCForm(self, IEN):
        X = self.mesh.X
        Y = self.mesh.Y

        Higher_IEN = {}

        for IENelem in tqdm(IEN.data):
            combinations = [[IENelem[1], IENelem[0]], [IENelem[2], IENelem[1]], [IENelem[0], IENelem[1]]]

            for combo in combinations:
                if combo in self.data:
                    Higher_IEN[str(100_000*combo[0] + combo[1])] = IENelem

        self.Higher_IEN = Higher_IEN

    def IENFunc(self) -> tuple :
        if self.data_type == 'Line':
            return self.IENLineFunc(self.data, self.data_tag, self.prescription)
        elif self.data_type == 'Tri':
            return self.IENTriFunc(self.data, self.data_tag, self.prescription)
        elif self.data_type == 'Tet':
            return self.IENTetFunc(self.data, self.data_tag, self.prescription)

    #form functions for 2D
    def IENTetFunc(self, IEN, BCType, prescription) -> tuple:
        X = self.mesh.X
        Y = self.mesh.Y
        Z = self.mesh.Z

        Npoints = X.shape[0]

        A = np.full(Npoints, None)
        b = np.full(Npoints, None)

        for idx, IENelem in enumerate(IEN):
            if BCType[idx] == 'Dirichlet':
                for point in IENelem:
                    if A[point] is None:   
                        temp = np.zeros(Npoints)
                        temp[point] = 1

                        A[point] = temp
                        b[point] = prescription[point]
                    else:
                        A[point, point] += 1
                        b[point] += prescription[point]
            elif BCType[idx] == 'Neumann':
                idx1 = IENelem[0]
                idx2 = IENelem[1]

                vector = np.array([ (X[idx2] - X[idx1]),( Y[idx2] - Y[idx1]), 0])
                z_unit = np.array([0, 0, 1])

                normal = np.cross(z_unit, vector)
                normal = normal/np.linalg.norm(normal)

                print(normal)
            elif BCType[idx] == 'Robin':
                pass
            
        return A,b

    #form functions for 2D
    def IENTriFunc(self, IEN, BCType, prescription) -> tuple:
        X = self.mesh.X
        Y = self.mesh.Y

        Npoints = X.shape[0]

        A = np.full(Npoints, None)
        b = np.full(Npoints, None)

        for idx, IENelem in enumerate(IEN):
            if BCType[idx] == 'Dirichlet':
                for point in IENelem:
                    if A[point] is None:   
                        temp = np.zeros(Npoints)
                        temp[point] = 1

                        A[point] = temp
                        b[point] = prescription[point]
                    else:
                        A[point, point] += 1
                        b[point] += prescription[point]
            elif BCType[idx] == 'Neumann':
                idx1 = IENelem[0]
                idx2 = IENelem[1]

                vector = np.array([ (X[idx2] - X[idx1]),( Y[idx2] - Y[idx1]), 0])
                z_unit = np.array([0, 0, 1])

                normal = np.cross(z_unit, vector)
                normal = normal/np.linalg.norm(normal)
            elif BCType[idx] == 'Robin':
                pass
            
        return A,b

    #form functions for 2D
    def IENLineFunc(self, IEN, BCType, prescription) -> tuple:
        X = self.mesh.X
        Y = self.mesh.Y

        Npoints = X.shape[0]

        A = np.full(Npoints, None)
        b = np.full(Npoints, None)

        for idx, IENelem in enumerate(IEN):
            if BCType[idx] == 'Dirichlet':
                for point in IENelem:
                    if A[point] is None:   
                        A[point] = np.zeros(Npoints)
                        b[point] = 0

                    A[point][point] += 1
                    b[point] += prescription[idx]['T']
            else:
                idx1 = IENelem[0]
                idx2 = IENelem[1]

                vector = np.array([ (X[idx2] - X[idx1]),( Y[idx2] - Y[idx1]), 0])
                z_unit = np.array([0, 0, 1])

                normal = np.cross(z_unit, vector)
                normal = normal/np.linalg.norm(normal)

                Higher_IEN = self.Higher_IEN[str(100_000*IENelem[0] + IENelem[1])]
                tri_matrix = [[1, X[Higher_IEN[i]], Y[Higher_IEN[i]]] for i in range(3)]

                tri_area = (1/2)*np.linalg.det(tri_matrix)

                [a_list, b_list, c_list] = np.linalg.inv(tri_matrix)*(2*tri_area)

                gxelem = (1/6)*b_list
                gyelem = (1/6)*c_list


                directed_flux_matrix = normal[0]*gxelem + normal[1]*gyelem


                for iglobal in IENelem:
                    if A[iglobal] is None:   
                        A[iglobal] = np.zeros(Npoints)
                        b[iglobal] = 0

                    for jlocal in range(3):
                        jglobal = Higher_IEN[jlocal]


                        A[iglobal][jglobal] += directed_flux_matrix[jlocal]

                    if BCType[idx] == 'Neumann':
                        b[iglobal] += prescription[idx]['q']
                    elif BCType[idx] == 'Robin':
                        A[iglobal][iglobal] += prescription[idx]['h']
                        b[iglobal] += prescription[idx]['h']*prescription[idx]['Ta']
            
        return A,b
   
class Mesh_class():
    def __init__(self, X, Y, Z, bc_idx, core_idx) -> None:
        self.X = X

        if len(Y) == 0:
            self.Y = np.zeros(X.shape[0])
        else:
            self.Y = Y

        if len(Z) == 0:
            self.Z = np.zeros(X.shape[0])
        else:
            self.Z = Z

        self.bc_idx = bc_idx  
        self.core_idx = core_idx 